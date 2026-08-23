#!/usr/bin/env python3
"""
Qwen3.8-27B-FP8 单卡交互式推理脚本（已修复 gate_proj FP8 scale 丢失）。

默认功能：
- 本地加载 /root/ai-models/Qwen3.8-27B-FP8。
- 修复 modules_to_not_convert 中未锚定的 *.mlp.gate 规则。
- 确认 64 层 gate_proj 都是 FP8Linear，且 weight_scale_inv 已加载。
- 支持文本以及本地图片/图片 URL 输入。
- 可选启用内置模拟天气工具，测试完整 Function Call 闭环。
- 第一轮打印 Prefill / Hidden States / KV Cache / 显存信息。
- 生成过程可逐 token 继续，也可一键自动生成到 EOS。
- 一轮结束后可继续输入新问题，并在 token 前缀一致时跨轮复用 KV Cache。

生成时的命令：
  Enter  生成下一个 token
  a      自动生成到 EOS 或 max-new-tokens
  q      停止当前回答

启动时使用 --enable-tools 启用内置 Function Call 测试，默认不携带工具定义。

对话时的命令：
  /image <路径或URL> [问题]  发送图片；包含空格的路径请加引号
  /clear                    清空对话历史
  /help                     显示帮助
  /exit                     退出程序
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)


DEFAULT_MODEL_PATH = "/root/ai-models/Qwen3.8-27B-FP8"
DEFAULT_IMAGE_PROMPT = "请描述这张图片。"
MAX_TOOL_ROUNDS = 4

TEST_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_mock_weather",
            "description": "返回指定城市的模拟天气数据，仅用于测试工具调用流程，不代表真实实时天气。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如北京、上海或杭州。",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认使用摄氏度。",
                    },
                },
                "required": ["city"],
            },
        },
    }
]


@dataclass
class GenerationResult:
    clean_text: str
    raw_text: str
    message_text: str
    sequences: torch.LongTensor
    past_key_values: Any


@dataclass
class ConversationKVCache:
    """同时保存模型 Cache 和该 Cache 实际覆盖的精确 token 前缀。"""

    past_key_values: Any = None
    cached_prefix_ids: torch.LongTensor | None = None

    @property
    def length(self) -> int:
        if self.cached_prefix_ids is None:
            return 0
        return int(self.cached_prefix_ids.shape[1])

    def reset(self) -> None:
        self.past_key_values = None
        self.cached_prefix_ids = None


def configure_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def section(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}", flush=True)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception as exc:
        return f"执行失败: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.8-27B-FP8 交互式推理")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="本地模型目录")
    parser.add_argument("--device", default="cuda:0", help="运行设备，默认 cuda:0")
    parser.add_argument(
        "--prompt",
        default="",
        help="可选的第一轮问题；默认留空并在模型加载后等待用户输入",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="第一轮图片的本地路径或 HTTP(S) URL；可重复指定以输入多张图片",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256, help="每轮最大生成 token 数")
    parser.add_argument("--enable-thinking", action="store_true", help="开启 thinking")
    parser.add_argument(
        "--enable-tools",
        action="store_true",
        help="携带内置工具定义，启用 Function Call 的生成、执行与结果回传",
    )
    parser.add_argument("--auto", action="store_true", help="从第一个 token 开始自动生成")
    parser.add_argument(
        "--verbose-tokens",
        action="store_true",
        help="打印每一步的 token ID/文本及最终原始 token；默认仅流式输出新增文本",
    )
    parser.add_argument("--one-shot", action="store_true", help="第一轮回答完成后退出")
    parser.add_argument(
        "--skip-prefill-inspection",
        action="store_true",
        help="跳过第一轮 Prefill/Hidden States 检查",
    )
    return parser.parse_args()


def print_environment() -> None:
    section("环境信息")
    print("Python:", sys.version.replace("\n", " "))
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print("Transformers:", transformers.__version__)
    print("Accelerate:", package_version("accelerate"))
    print("Pillow:", package_version("Pillow"))
    print("kernels:", package_version("kernels"))
    print("flash-linear-attention:", package_version("flash-linear-attention"))
    print("causal-conv1d:", package_version("causal-conv1d"))

    nvcc = shutil.which("nvcc")
    print("nvcc:", nvcc or "未找到")
    if nvcc:
        print(command_output([nvcc, "--version"]))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，FP8 模型需要 CUDA GPU")

    print("GPU 数量:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {props.name}, "
            f"{props.total_memory / 1024**3:.2f} GiB, "
            f"Compute Capability {props.major}.{props.minor}"
        )

    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (8, 9):
        raise RuntimeError(
            f"当前 GPU Compute Capability={major}.{minor}，"
            "Transformers FineGrained FP8 需要 >= 8.9"
        )

    if package_version("kernels") == "未安装":
        raise RuntimeError(
            "缺少 FP8 kernels 包。请执行: "
            "python -m pip install --no-cache-dir kernels==0.15.2"
        )

    if package_version("flash-linear-attention") == "未安装" or package_version("causal-conv1d") == "未安装":
        print(
            "[提示] 线性注意力快速路径不完整，Transformers 将回退到 PyTorch。\n"
            "       这会降低速度、增加显存，但不影响结果正确性。"
        )


def quant_config_get(quant_config: Any, name: str, default: Any = None) -> Any:
    if isinstance(quant_config, dict):
        return quant_config.get(name, default)
    return getattr(quant_config, name, default)


def quant_config_set(quant_config: Any, name: str, value: Any) -> None:
    if isinstance(quant_config, dict):
        quant_config[name] = value
    else:
        setattr(quant_config, name, value)


def load_and_fix_config(model_path: Path) -> tuple[Any, int]:
    section("加载并修复 Config")
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    expected_layers = int(getattr(text_config, "num_hidden_layers"))

    print("Config 类型:", type(config).__name__)
    print("model_type:", getattr(config, "model_type", None))
    print("文本层数:", expected_layers)
    vision_config = getattr(config, "vision_config", None)
    print("视觉 Config:", type(vision_config).__name__ if vision_config is not None else "无")

    quant_config = getattr(config, "quantization_config", None)
    if quant_config is None:
        raise RuntimeError("config.json 中没有 quantization_config，请确认下载的是 FP8 模型")

    quant_method = quant_config_get(quant_config, "quant_method", "未知")
    block_size = quant_config_get(quant_config, "weight_block_size", "未知")
    modules = list(quant_config_get(quant_config, "modules_to_not_convert", []) or [])
    print("量化方法:", quant_method)
    print("权重 block size:", block_size)
    print("modules_to_not_convert 数量:", len(modules))

    fixed_modules: list[str] = []
    changed: list[tuple[str, str]] = []
    for original in modules:
        original = str(original)
        if original.endswith("mlp.gate") and not original.endswith("mlp.gate$"):
            # 只增加结尾锚点。不要 re.escape 整条规则，
            # 否则可能破坏 checkpoint 原本的正则/通配语义。
            fixed = original + "$"
            changed.append((original, fixed))
            fixed_modules.append(fixed)
        else:
            fixed_modules.append(original)

    quant_config_set(quant_config, "modules_to_not_convert", fixed_modules)

    print(f"修复的 mlp.gate 规则: {len(changed)} 条")
    for before, after in changed[:3]:
        print(f"  {before!r} -> {after!r}")
    if len(changed) > 6:
        print("  ...")
    for before, after in changed[-3:]:
        if (before, after) not in changed[:3]:
            print(f"  {before!r} -> {after!r}")

    if not changed:
        already_fixed = sum(str(item).endswith("mlp.gate$") for item in modules)
        if already_fixed:
            print(f"[OK] Config 中已有 {already_fixed} 条锚定后的 mlp.gate 规则")
        else:
            print("[提示] 没有发现需要修复的 mlp.gate 规则")
    else:
        print("[OK] 修复只存在于当前进程内存，不会改写模型 config.json")

    return config, expected_layers


def load_processor_and_model(
    model_path: Path,
    config: Any,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    section("加载 Processor")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    print("Processor:", type(processor).__name__)

    section("加载 FP8 模型")
    loaded = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        output_loading_info=True,
    )

    if isinstance(loaded, tuple) and len(loaded) == 2:
        model, loading_info = loaded
    else:
        model, loading_info = loaded, {}

    model.eval()
    print("模型类型:", type(model).__name__)
    print("模型设备:", model.device)
    print("模型默认 dtype:", model.dtype)

    missing = list(loading_info.get("missing_keys", []) or [])
    unexpected = list(loading_info.get("unexpected_keys", []) or [])
    mismatched = list(loading_info.get("mismatched_keys", []) or [])
    print("missing_keys:", len(missing))
    print("unexpected_keys:", len(unexpected))
    print("mismatched_keys:", len(mismatched))

    discarded_gate_scales = [
        str(key) for key in unexpected if ".mlp.gate_proj.weight_scale_inv" in str(key)
    ]
    if discarded_gate_scales:
        raise RuntimeError(
            f"仍有 {len(discarded_gate_scales)} 个 gate_proj.weight_scale_inv 被丢弃: "
            f"{discarded_gate_scales[:3]}"
        )

    if unexpected:
        print("unexpected_keys 示例:")
        for key in unexpected[:8]:
            print(" ", key)

    return processor, model, loading_info


def verify_gate_proj(model: Any, expected_layers: int) -> None:
    section("gate_proj FP8 完整性检查")
    modules = [(name, module) for name, module in model.named_modules() if name.endswith(".mlp.gate_proj")]

    missing_scales: list[str] = []
    wrong_types: list[str] = []
    invalid_scales: list[str] = []
    wrong_weight_dtype: list[str] = []

    for name, module in modules:
        if type(module).__name__ != "FP8Linear":
            wrong_types.append(name)

        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.float8_e4m3fn:
            wrong_weight_dtype.append(name)

        scale = getattr(module, "weight_scale_inv", None)
        if not isinstance(scale, torch.Tensor):
            missing_scales.append(name)
            continue

        scale_f32 = scale.detach().float()
        if not bool(torch.isfinite(scale_f32).all().item()) or not bool((scale_f32 > 0).all().item()):
            invalid_scales.append(name)

    print(f"gate_proj 数量: {len(modules)} / expected {expected_layers}")
    print("FP8Linear 数量:", len(modules) - len(wrong_types))
    print("float8_e4m3fn 权重数量:", len(modules) - len(wrong_weight_dtype))
    print("缺失 weight_scale_inv:", len(missing_scales))
    print("异常 weight_scale_inv:", len(invalid_scales))

    if len(modules) != expected_layers:
        raise RuntimeError(f"gate_proj 数量不正确: {len(modules)} != {expected_layers}")
    if wrong_types or wrong_weight_dtype or missing_scales or invalid_scales:
        raise RuntimeError(
            "gate_proj FP8 加载仍不正常: "
            f"wrong_types={len(wrong_types)}, "
            f"wrong_dtype={len(wrong_weight_dtype)}, "
            f"missing_scales={len(missing_scales)}, "
            f"invalid_scales={len(invalid_scales)}"
        )

    for name, module in modules[:1] + modules[-1:]:
        scale = module.weight_scale_inv.detach().float()
        print(
            f"  {name}: type={type(module).__name__}, weight={module.weight.dtype}, "
            f"scale_shape={tuple(scale.shape)}, "
            f"scale_min={scale.min().item():.6g}, scale_max={scale.max().item():.6g}"
        )

    print("[OK] 64 层 gate_proj 的 FP8 权重和 scale 均已正确加载")


def get_input_device(model: Any, fallback: str) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device(fallback)


def apply_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    enable_thinking: bool,
    device: torch.device,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    template_kwargs: dict[str, Any] = {
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
        # 历史 assistant 的 think 块必须保持稳定，否则下一轮 token 前缀会变化。
        "preserve_thinking": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if tools is not None:
        template_kwargs["tools"] = tools

    inputs = processor.apply_chat_template(messages, **template_kwargs)

    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def cache_sequence_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if not callable(get_seq_length):
        raise TypeError(f"当前 Cache 类型不支持 get_seq_length(): {type(past_key_values).__name__}")
    return int(get_seq_length())


def reset_model_cache_state(model: Any) -> None:
    """清除 Qwen 多模态位置编码中与旧 Cache 绑定的派生状态。"""
    model_core = getattr(model, "model", None)
    if model_core is not None and hasattr(model_core, "rope_deltas"):
        model_core.rope_deltas = None


def reset_conversation_cache(cache_state: ConversationKVCache, model: Any, reason: str) -> None:
    if cache_state.past_key_values is not None:
        print(f"[KV Cache] 已重置: {reason}")
    cache_state.reset()
    reset_model_cache_state(model)


def prepare_inputs_with_conversation_cache(
    full_inputs: dict[str, Any],
    cache_state: ConversationKVCache,
    model: Any,
    allow_reuse: bool,
) -> tuple[dict[str, Any], bool]:
    """校验完整 Prompt 前缀，让 Transformers 根据 Cache 长度自动切出新增 token。"""
    if cache_state.past_key_values is None:
        return full_inputs, False

    if not allow_reuse:
        reset_conversation_cache(cache_state, model, "当前轮包含新图片，需要重新编码视觉输入")
        return full_inputs, False

    try:
        actual_cache_length = cache_sequence_length(cache_state.past_key_values)
    except (TypeError, ValueError) as exc:
        reset_conversation_cache(cache_state, model, str(exc))
        return full_inputs, False

    cached_prefix_ids = cache_state.cached_prefix_ids
    if cached_prefix_ids is None or actual_cache_length != cache_state.length:
        reset_conversation_cache(
            cache_state,
            model,
            f"Cache 长度和已记录 token 不一致: cache={actual_cache_length}, token={cache_state.length}",
        )
        return full_inputs, False

    full_input_ids = full_inputs["input_ids"]
    full_length = int(full_input_ids.shape[1])
    if full_length <= actual_cache_length:
        reset_conversation_cache(
            cache_state,
            model,
            f"新 Prompt 长度 {full_length} 没有超过 Cache 长度 {actual_cache_length}",
        )
        return full_inputs, False

    expected_prefix = cached_prefix_ids.to(full_input_ids.device)
    actual_prefix = full_input_ids[:, :actual_cache_length]
    if expected_prefix.shape != actual_prefix.shape or not torch.equal(expected_prefix, actual_prefix):
        reset_conversation_cache(cache_state, model, "聊天模板或历史 token 前缀发生变化")
        return full_inputs, False

    generation_inputs: dict[str, Any] = {
        # input_ids 和 attention_mask 都保留完整长度。GenerationMixin 检测到
        # past_key_values 后，会在 Prefill 内部自动切出 cache 之后的新增 token。
        "input_ids": full_input_ids,
        "attention_mask": full_inputs["attention_mask"],
        "past_key_values": cache_state.past_key_values,
    }
    # 类型和位置字段也保留完整长度，GenerationMixin 会和 input_ids 一起切片。
    # 不复制 pixel_values / image_grid_thw 等视觉字段：历史图片已经进入 Cache，
    # 再传视觉张量会与内部切片后的 input_ids 不匹配。
    for name in ("token_type_ids", "mm_token_type_ids", "position_ids"):
        if name in full_inputs:
            generation_inputs[name] = full_inputs[name]

    continuation_length = full_length - actual_cache_length
    print(
        f"[KV Cache] 命中: 复用 {actual_cache_length} token，"
        f"Transformers 将只 Prefill {continuation_length} 个新增 token"
        f"（传入的完整 Prompt 为 {full_length} token）"
    )
    return generation_inputs, True


def log_model_input_token_segments(
    tokenizer: Any,
    full_input_ids: torch.LongTensor,
    cached_length: int,
) -> None:
    """以可读文本显示完整模型输入，并标记已缓存与本次新增的 token 区间。"""
    token_ids = full_input_ids[0].detach().cpu().tolist()
    total_length = len(token_ids)
    cached_length = max(0, min(cached_length, total_length))

    def decode_span(start: int, end: int) -> str:
        if start >= end:
            return "（无）"
        return tokenizer.decode(
            token_ids[start:end],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    section("本次模型输入 Token 分区")
    print(
        f"总计 {total_length} token：KV Cache 前缀 {cached_length} token，"
        f"本次新增 Prefill {total_length - cached_length} token"
    )
    print("\n----- KV Cache 已缓存前缀（这些 token 对应的 K/V 不会重复计算）-----")
    print(decode_span(0, cached_length))
    print("\n----- 本次新增输入（模型实际执行 Prefill 的 token）-----")
    print(decode_span(cached_length, total_length), flush=True)


def update_conversation_cache(
    cache_state: ConversationKVCache,
    generation_result: GenerationResult,
    model: Any,
) -> None:
    """用本次实际送入 generate() 的序列补齐精确 Cache token 前缀。"""
    new_cache = generation_result.past_key_values
    if new_cache is None:
        reset_conversation_cache(cache_state, model, "generate() 没有返回 past_key_values")
        return

    old_length = cache_state.length
    try:
        new_length = cache_sequence_length(new_cache)
    except (TypeError, ValueError) as exc:
        reset_conversation_cache(cache_state, model, str(exc))
        return

    sequence_length = int(generation_result.sequences.shape[1])
    if new_length < 0 or new_length > sequence_length:
        reset_conversation_cache(
            cache_state,
            model,
            f"无法对齐 Cache: cache={new_length}, sequence={sequence_length}",
        )
        return

    cache_state.past_key_values = new_cache
    cache_state.cached_prefix_ids = generation_result.sequences[:, :new_length].detach().clone()
    print(
        f"[KV Cache] 已更新: {old_length} -> {new_length} token，"
        f"当前完整序列仍有 {sequence_length - new_length} 个末尾 token 尚未写入 Cache"
    )


def inspect_prefill_once(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    verbose_tokens: bool,
) -> None:
    section("第一轮 Prefill 检查")
    input_ids = inputs["input_ids"]
    print("input_ids shape:", tuple(input_ids.shape))
    if verbose_tokens:
        print("token ids:", input_ids[0].tolist())
    print("输入字段:")
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

    print("输出类型:", type(outputs).__name__)
    print("logits shape:", tuple(outputs.logits.shape))
    print("logits all finite:", bool(torch.isfinite(outputs.logits).all().item()))
    print("hidden states 数量:", len(outputs.hidden_states))
    print("第 0 层输入:", tuple(outputs.hidden_states[0].shape))
    print("最后一层输出:", tuple(outputs.hidden_states[-1].shape))
    print("cache 类型:", type(outputs.past_key_values).__name__)
    if hasattr(outputs.past_key_values, "get_seq_length"):
        print("cache 当前序列长度:", outputs.past_key_values.get_seq_length())

    next_token_id = int(outputs.logits[:, -1].argmax(dim=-1).item())
    print("贪心首 token id:", next_token_id)
    print("贪心首 token:", repr(processor.tokenizer.decode([next_token_id])))
    print(f"Prefill 峰值显存: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

    del outputs
    torch.cuda.empty_cache()


class InteractiveTokenController(StoppingCriteria):
    """
    在 model.generate() 内部每生成一个 token 后暂停。

    这样 KV Cache、cache_position、attention_mask 和 Qwen M-RoPE 仍由
    Transformers GenerationMixin 负责，不需要手工维护内部状态。
    """

    def __init__(
        self,
        tokenizer: Any,
        prompt_length: int,
        eos_token_ids: set[int],
        start_auto: bool,
        verbose_tokens: bool,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.last_length = prompt_length
        self.eos_token_ids = eos_token_ids
        self.auto = start_auto
        self.verbose_tokens = verbose_tokens
        self.stopped_by_user = False
        self.reached_eos = False
        self.step = 0

    @staticmethod
    def _decision(batch_size: int, device: torch.device, stop: bool) -> torch.BoolTensor:
        return torch.full((batch_size,), stop, dtype=torch.bool, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        device = input_ids.device
        current_length = int(input_ids.shape[1])

        if current_length <= self.last_length:
            return self._decision(batch_size, device, False)

        for position in range(self.last_length, current_length):
            token_id = int(input_ids[0, position].item())
            self.step += 1
            if self.verbose_tokens:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                print(f"\n[Step {self.step}] token_id={token_id}, token={token_text!r}", flush=True)

        self.last_length = current_length
        latest_id = int(input_ids[0, -1].item())
        if latest_id in self.eos_token_ids:
            self.reached_eos = True
            return self._decision(batch_size, device, True)

        if self.auto:
            return self._decision(batch_size, device, False)

        while True:
            try:
                command = input("\n[Enter=下一个, a=自动运行, q=停止当前回答] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n已停止当前回答。")
                self.stopped_by_user = True
                return self._decision(batch_size, device, True)

            if command == "":
                return self._decision(batch_size, device, False)
            if command == "a":
                self.auto = True
                return self._decision(batch_size, device, False)
            if command == "q":
                self.stopped_by_user = True
                return self._decision(batch_size, device, True)
            print("无效命令，请输入 Enter、a 或 q。")


def normalize_eos_token_ids(tokenizer: Any, model: Any) -> set[int]:
    candidates = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
    ]
    result: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, (list, tuple, set)):
            result.update(int(item) for item in candidate)
        else:
            result.add(int(candidate))
    return result


def split_assistant_message(message_text: str, enable_thinking: bool) -> tuple[str, str]:
    """拆分 Qwen 模板中的 reasoning_content 和最终 content，便于精确重建历史。"""
    marker_start = "<think>\n"
    marker_end = "\n</think>\n\n"
    if message_text.startswith(marker_start) and marker_end in message_text:
        reasoning_and_content = message_text[len(marker_start) :]
        reasoning_content, content = reasoning_and_content.split(marker_end, 1)
        return reasoning_content.strip(), content.strip()

    # enable_thinking=True 时，generation prompt 已经以 ``<think>\n`` 结尾，
    # 所以 generate() 返回的增量序列不会再次包含开标签，只包含思考正文和闭标签。
    if enable_thinking:
        if marker_end in message_text:
            reasoning_content, content = message_text.split(marker_end, 1)
            return reasoning_content.strip(), content.strip()
        if "</think>" in message_text:
            reasoning_content, content = message_text.split("</think>", 1)
            return reasoning_content.strip(), content.strip()
        # 用户中止或 max_new_tokens 截断在思考阶段时，也要把已有 token
        # 记录为 reasoning_content，保证下一轮重新套模板后的 token 前缀不变。
        return message_text.strip(), ""

    match = re.match(r"^\s*<think>\s*(.*?)\s*</think>\s*(.*)$", message_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", message_text.strip()


def generate_one_response(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    max_new_tokens: int,
    start_auto: bool,
    verbose_tokens: bool,
) -> GenerationResult:
    section("开始交互式 Decode")
    if start_auto:
        print("自动生成中；按 Ctrl+C 可退出程序。")
    else:
        print("操作方式：Enter=下一个 token，a=自动生成，q=停止当前回答")

    input_length = int(inputs["input_ids"].shape[1])
    eos_token_ids = normalize_eos_token_ids(processor.tokenizer, model)
    controller = InteractiveTokenController(
        tokenizer=processor.tokenizer,
        prompt_length=input_length,
        eos_token_ids=eos_token_ids,
        start_auto=start_auto,
        verbose_tokens=verbose_tokens,
    )
    streamer = None
    if not verbose_tokens:
        print("\n模型> ", end="", flush=True)
        streamer = TextStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            stopping_criteria=StoppingCriteriaList([controller]),
            streamer=streamer,
        )

    generated_ids = generation_output.sequences
    new_ids = generated_ids[0, input_length:]
    raw_text = processor.tokenizer.decode(new_ids, skip_special_tokens=False)
    clean_text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    message_end = int(new_ids.numel())
    while message_end > 0 and int(new_ids[message_end - 1].item()) in eos_token_ids:
        message_end -= 1
    message_text = processor.tokenizer.decode(
        new_ids[:message_end],
        skip_special_tokens=False,
    )
    peak_memory_gib = torch.cuda.max_memory_allocated() / 1024**3
    if verbose_tokens:
        section("当前回答结果")
        print("生成 token 数:", int(new_ids.numel()))
        print("生成 token ids:", new_ids.tolist())
        print("保留 special tokens:")
        print(raw_text)
        print("最终文本:")
        print(clean_text)
        print(f"峰值显存: {peak_memory_gib:.2f} GiB")
    else:
        print(f"[完成] 生成 {int(new_ids.numel())} token，峰值显存 {peak_memory_gib:.2f} GiB")

    if controller.stopped_by_user:
        print("[提示] 当前回答由用户提前停止。")
    elif controller.reached_eos:
        print("[OK] 当前回答正常到达 EOS。")
    elif int(new_ids.numel()) >= max_new_tokens:
        print("[提示] 当前回答到达 max_new_tokens 限制。")

    if len(new_ids) >= 8:
        token_id, count = Counter(new_ids.tolist()).most_common(1)[0]
        if count >= max(8, len(new_ids) // 2):
            token_text = processor.tokenizer.decode([token_id], skip_special_tokens=False)
            print(f"[警告] 发现异常重复 token: id={token_id}, token={token_text!r}, count={count}")

    return GenerationResult(
        clean_text=clean_text,
        raw_text=raw_text,
        message_text=message_text,
        sequences=generated_ids,
        past_key_values=generation_output.past_key_values,
    )


def normalize_image_content(source: str) -> dict[str, str]:
    source = source.strip()
    if not source:
        raise ValueError("图片路径或 URL 不能为空")

    if source.startswith(("http://", "https://")):
        return {"type": "image", "url": source}

    image_path = Path(source).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    return {"type": "image", "path": str(image_path)}


def parse_image_command(command: str) -> tuple[dict[str, str], str]:
    raw_arguments = command[len("/image") :].strip()
    try:
        arguments = shlex.split(raw_arguments)
    except ValueError as exc:
        raise ValueError(f"无法解析 /image 命令: {exc}") from exc

    if not arguments:
        raise ValueError('用法: /image <图片路径或URL> [问题]；路径含空格时请写成 /image "/path/a b.jpg" 问题')

    image_content = normalize_image_content(arguments[0])
    question = " ".join(arguments[1:]).strip() or DEFAULT_IMAGE_PROMPT
    return image_content, question


def append_multimodal_user_message(
    messages: list[dict[str, Any]],
    text: str,
    images: list[dict[str, str]],
) -> None:
    content: list[dict[str, str]] = [dict(image) for image in images]
    content.append({"type": "text", "text": text})
    messages.append(
        {
            "role": "user",
            "content": content,
        }
    )


def append_assistant_message(
    messages: list[dict[str, Any]],
    text: str,
    reasoning_content: str = "",
) -> None:
    messages.append(
        {
            "role": "assistant",
            "reasoning_content": reasoning_content,
            "content": [{"type": "text", "text": text}],
        }
    )


TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL,
)
TOOL_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    flags=re.DOTALL,
)


def parse_tool_argument(raw_value: str) -> Any:
    value = raw_value.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """解析 Qwen3.8 chat_template 约定的 qwen3_coder XML 风格工具调用。"""
    tool_calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_PATTERN.finditer(text):
        function_name = match.group(1).strip()
        function_body = match.group(2)
        arguments = {
            parameter.group(1).strip(): parse_tool_argument(parameter.group(2))
            for parameter in TOOL_PARAMETER_PATTERN.finditer(function_body)
        }
        tool_calls.append(
            {
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": arguments,
                },
            }
        )
    return tool_calls


def execute_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """只执行显式列入白名单的本地测试工具。"""
    function = tool_call.get("function", {})
    function_name = function.get("name")
    arguments = function.get("arguments", {})

    if function_name != "get_mock_weather":
        return {
            "ok": False,
            "error": f"未知或不允许执行的工具: {function_name}",
        }
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "工具参数必须是对象"}

    city = arguments.get("city")
    unit = arguments.get("unit", "celsius")
    if not isinstance(city, str) or not city.strip():
        return {"ok": False, "error": "city 必须是非空字符串"}
    if unit not in {"celsius", "fahrenheit"}:
        return {"ok": False, "error": "unit 只能是 celsius 或 fahrenheit"}

    city = city.strip()
    weather_by_city = {
        "北京": (26.0, "晴"),
        "上海": (29.0, "多云"),
        "杭州": (28.0, "小雨"),
    }
    temperature, condition = weather_by_city.get(city, (25.0, "晴间多云"))
    if unit == "fahrenheit":
        temperature = round(temperature * 9 / 5 + 32, 1)

    return {
        "ok": True,
        "mock": True,
        "city": city,
        "temperature": temperature,
        "unit": unit,
        "condition": condition,
        "notice": "这是本地固定的模拟数据，仅用于 Function Call 流程测试。",
    }


def append_assistant_tool_calls(
    messages: list[dict[str, Any]],
    content: str,
    tool_calls: list[dict[str, Any]],
    reasoning_content: str = "",
) -> None:
    messages.append(
        {
            "role": "assistant",
            "reasoning_content": reasoning_content,
            "content": content,
            "tool_calls": tool_calls,
        }
    )


def append_tool_result(messages: list[dict[str, Any]], result: dict[str, Any]) -> None:
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(result, ensure_ascii=False),
        }
    )


def run_chat_loop(
    model: Any,
    processor: Any,
    device: torch.device,
    first_prompt: str,
    first_images: list[str],
    max_new_tokens: int,
    enable_thinking: bool,
    start_auto: bool,
    one_shot: bool,
    inspect_prefill: bool,
    enable_tools: bool,
    verbose_tokens: bool,
) -> None:
    messages: list[dict[str, Any]] = []
    cache_state = ConversationKVCache()
    current_prompt = first_prompt.strip()
    current_images = [normalize_image_content(source) for source in first_images]
    first_round = True

    section("进入对话")
    print("enable_thinking:", enable_thinking)
    print("enable_tools:", enable_tools)
    print("conversation_kv_cache: enabled（新增图片时自动回退完整 Prefill）")
    print("token_output:", "verbose" if verbose_tokens else "streaming")
    print("max_new_tokens:", max_new_tokens)
    print("对话命令: /image <路径或URL> [问题], /clear, /help, /exit")
    if enable_tools:
        print("可用测试工具: get_mock_weather（返回本地模拟数据）")

    while True:
        if not current_prompt:
            try:
                current_prompt = input("\n用户> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                return

        if current_prompt.lower() in {"/exit", "exit", "quit"}:
            print("退出。")
            return
        if current_prompt.lower() == "/clear":
            messages.clear()
            reset_conversation_cache(cache_state, model, "用户执行 /clear")
            current_images = []
            print("[OK] 已清空对话历史。")
            current_prompt = ""
            continue

        if current_prompt.lower() == "/help":
            print('  /image <路径或URL> [问题]  例如: /image "/data/my photo.jpg" 图片里有什么？')
            if enable_tools:
                print("  工具测试示例: 北京现在天气怎么样？请调用工具查询。")
            print("  /clear                    清空对话历史")
            print("  /exit                     退出程序")
            current_prompt = ""
            continue

        if current_prompt.lower() == "/image" or current_prompt.lower().startswith("/image "):
            try:
                image_content, image_question = parse_image_command(current_prompt)
            except (ValueError, FileNotFoundError) as exc:
                print(f"[错误] {exc}")
                current_prompt = ""
                continue
            current_images = [image_content]
            current_prompt = image_question

        if current_images:
            if getattr(model.config, "vision_config", None) is None:
                raise RuntimeError("当前 checkpoint 没有 vision_config，不能接收图片")
            if getattr(processor, "image_processor", None) is None:
                raise RuntimeError("当前 Processor 没有 image_processor，不能预处理图片")
            if not callable(getattr(model, "get_image_features", None)):
                raise RuntimeError("当前模型类没有 get_image_features()，不能提取图片特征")

        print(f"\n用户> {current_prompt}")
        for image in current_images:
            print("图片>", image.get("path") or image.get("url"))
        append_multimodal_user_message(messages, current_prompt, current_images)
        full_inputs = apply_chat_template(
            processor=processor,
            messages=messages,
            enable_thinking=enable_thinking,
            device=device,
            tools=TEST_TOOLS if enable_tools else None,
        )
        inputs, cache_reused = prepare_inputs_with_conversation_cache(
            full_inputs=full_inputs,
            cache_state=cache_state,
            model=model,
            allow_reuse=not current_images,
        )
        log_model_input_token_segments(
            tokenizer=processor.tokenizer,
            full_input_ids=full_inputs["input_ids"],
            cached_length=cache_state.length if cache_reused else 0,
        )

        if first_round and inspect_prefill:
            inspect_prefill_once(
                model,
                processor,
                full_inputs,
                verbose_tokens=verbose_tokens,
            )

        tool_round = 0
        while True:
            generation_result = generate_one_response(
                model=model,
                processor=processor,
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                start_auto=start_auto,
                verbose_tokens=verbose_tokens,
            )
            update_conversation_cache(cache_state, generation_result, model)
            # Cache 的唯一长期所有者应是 cache_state，避免 /clear 或图片回退后仍被旧结果引用。
            generation_result.past_key_values = None
            reasoning_content, assistant_message_text = split_assistant_message(
                generation_result.message_text,
                enable_thinking=enable_thinking,
            )
            tool_calls = parse_tool_calls(generation_result.raw_text) if enable_tools else []

            if not tool_calls:
                if enable_tools and "<tool_call>" in generation_result.raw_text:
                    print("[警告] 检测到工具调用标记，但格式不完整，已作为普通回答保留。")
                append_assistant_message(
                    messages,
                    assistant_message_text,
                    reasoning_content=reasoning_content,
                )
                break

            if tool_round >= MAX_TOOL_ROUNDS:
                print(f"[警告] 已达到最大工具调用轮数 {MAX_TOOL_ROUNDS}，停止自动执行工具。")
                append_assistant_message(
                    messages,
                    assistant_message_text,
                    reasoning_content=reasoning_content,
                )
                break
            tool_round += 1

            assistant_content = assistant_message_text.split("<tool_call>", 1)[0].strip()
            append_assistant_tool_calls(
                messages,
                assistant_content,
                tool_calls,
                reasoning_content=reasoning_content,
            )

            section(f"执行工具（第 {tool_round} 轮）")
            for tool_call in tool_calls:
                function = tool_call["function"]
                print("工具名:", function["name"])
                print("参数:", json.dumps(function["arguments"], ensure_ascii=False))
                result = execute_tool_call(tool_call)
                print("结果:", json.dumps(result, ensure_ascii=False))
                append_tool_result(messages, result)

            full_inputs = apply_chat_template(
                processor=processor,
                messages=messages,
                enable_thinking=enable_thinking,
                device=device,
                tools=TEST_TOOLS,
            )
            inputs, cache_reused = prepare_inputs_with_conversation_cache(
                full_inputs=full_inputs,
                cache_state=cache_state,
                model=model,
                allow_reuse=True,
            )
            log_model_input_token_segments(
                tokenizer=processor.tokenizer,
                full_input_ids=full_inputs["input_ids"],
                cached_length=cache_state.length if cache_reused else 0,
            )

        # 下轮会重新构造输入，及时释放历史图片张量和旧的 generate() 参数字典。
        full_inputs = {}
        inputs = {}

        if one_shot:
            return

        first_round = False
        current_prompt = ""
        current_images = []


def main() -> int:
    configure_utf8_output()
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.is_dir():
        print(f"[错误] 模型目录不存在: {model_path}", file=sys.stderr)
        return 2

    try:
        print_environment()
        config, expected_layers = load_and_fix_config(model_path)
        processor, model, _ = load_processor_and_model(
            model_path=model_path,
            config=config,
            device=args.device,
        )
        verify_gate_proj(model, expected_layers)

        device = get_input_device(model, args.device)
        first_prompt = args.prompt
        run_chat_loop(
            model=model,
            processor=processor,
            device=device,
            first_prompt=first_prompt,
            first_images=args.image,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=args.enable_thinking,
            start_auto=args.auto,
            one_shot=args.one_shot,
            inspect_prefill=not args.skip_prefill_inspection,
            enable_tools=args.enable_tools,
            verbose_tokens=args.verbose_tokens,
        )
        return 0
    except KeyboardInterrupt:
        print("\n用户中断，退出。")
        return 130
    except Exception as exc:
        print(f"\n[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
