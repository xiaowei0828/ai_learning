#!/usr/bin/env python3
"""
Qwen3.8-27B-FP8 输出乱码排查脚本。

这个脚本会：
1. 检查 Python / PyTorch / CUDA / Transformers / FP8 kernel 环境。
2. 检查 checkpoint 中 gate_proj.weight_scale_inv 是否存在。
3. 检查并修复 modules_to_not_convert 中未锚定的 *.mlp.gate 规则，
   避免它误匹配 *.mlp.gate_proj。
4. 加载模型后检查每一层 gate_proj 的类型、FP8 权重和 scale。
5. 使用官方 model.generate() 执行确定性最小生成测试。

默认会修复 gate 规则。使用 --no-fix-gate-pattern 可重现未修复的加载行为。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "/root/ai-models/Qwen3.8-27B-FP8"
DEFAULT_PROMPT = "请用一句话解释什么是KV Cache。"


def section(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}", flush=True)


def ok(message: str) -> None:
    print(f"[OK] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def fail(message: str) -> None:
    print(f"[FAIL] {message}", flush=True)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"


def run_text_command(command: list[str]) -> str:
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


def configure_utf8_output() -> None:
    # convert_ids_to_tokens() 中的字节级 BPE 形式不等于终端编码乱码。
    # 这里仍强制 stdout/stderr 使用 UTF-8，排除终端编码干扰。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="排查 Qwen3.8-27B-FP8 随机/多语种乱码输出")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="本地模型目录")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="测试提示词")
    parser.add_argument("--device", default="cuda:0", help="模型设备，默认 cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="最大生成 token 数")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="开启 thinking。排查时默认关闭，减少变量。",
    )
    parser.add_argument(
        "--no-fix-gate-pattern",
        action="store_true",
        help="不修复 *.mlp.gate 规则，用于 A/B 对比。",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="只检查环境、config 和 checkpoint key，不加载 27B 模型。",
    )
    return parser.parse_args()


def get_quantization_config(config: Any) -> Any:
    return getattr(config, "quantization_config", None)


def qcfg_get(qcfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(qcfg, dict):
        return qcfg.get(key, default)
    return getattr(qcfg, key, default)


def qcfg_set(qcfg: Any, key: str, value: Any) -> None:
    if isinstance(qcfg, dict):
        qcfg[key] = value
    else:
        setattr(qcfg, key, value)


def regex_matches(pattern: str, module_name: str) -> bool:
    try:
        return re.search(pattern, module_name) is not None
    except re.error:
        # 某些旧配置把它当作普通前缀，而不是完整正则。
        return module_name.startswith(pattern)


def inspect_and_patch_gate_patterns(config: Any, apply_fix: bool) -> dict[str, Any]:
    qcfg = get_quantization_config(config)
    if qcfg is None:
        fail("config 中没有 quantization_config，这不像 FP8 checkpoint")
        return {"suspicious": [], "patched": []}

    print("量化方法:", qcfg_get(qcfg, "quant_method", "未知"))
    print("权重 block size:", qcfg_get(qcfg, "weight_block_size", "未知"))
    print("scale format:", qcfg_get(qcfg, "scale_fmt", "未知"))

    original = list(qcfg_get(qcfg, "modules_to_not_convert", []) or [])
    print(f"modules_to_not_convert 数量: {len(original)}")

    samples = (
        "model.language_model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.gate_proj",
        "language_model.layers.0.mlp.gate_proj",
    )
    suspicious: list[str] = []
    patched: list[str] = []

    for pattern in original:
        pattern = str(pattern)
        # 未带 $ 的 *.mlp.gate 会把 *.mlp.gate_proj 也当作忽略模块。
        is_unanchored_gate = pattern.endswith("mlp.gate") and not pattern.endswith("mlp.gate$")
        matches_gate_proj = any(regex_matches(pattern, sample) for sample in samples)
        if is_unanchored_gate or matches_gate_proj:
            suspicious.append(pattern)
            print(f"  [可疑] {pattern!r}  可能匹配 gate_proj={matches_gate_proj}")

        if apply_fix and is_unanchored_gate:
            # 仅增加结尾锚点；不能 re.escape 整个 pattern，否则会破坏原有层号通配符。
            patched.append(pattern + "$")
        else:
            patched.append(pattern)

    if suspicious:
        warn(f"发现 {len(suspicious)} 条可能误忽略 gate_proj 的规则")
    else:
        ok("未发现明显会误匹配 gate_proj 的忽略规则")

    if apply_fix:
        qcfg_set(qcfg, "modules_to_not_convert", patched)
        changes = [(before, after) for before, after in zip(original, patched) if before != after]
        if changes:
            for before, after in changes:
                print(f"  [已修复] {before!r} -> {after!r}")
            ok("已将修复后的量化配置写入本次内存中的 config")
        else:
            print("没有需要修复的 gate pattern")
    else:
        warn("已按参数跳过 gate pattern 修复")

    return {"original": original, "suspicious": suspicious, "patched": patched}


def checkpoint_keys_from_index(model_path: Path) -> tuple[set[str], dict[str, str]]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return set(), {}
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map", {})
    return set(weight_map), dict(weight_map)


def checkpoint_keys_from_safetensors(model_path: Path) -> set[str]:
    try:
        from safetensors import safe_open
    except ImportError:
        warn("未安装 safetensors，无法扫描无 index 的 checkpoint")
        return set()

    keys: set[str] = set()
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys.update(handle.keys())
    return keys


def inspect_checkpoint(model_path: Path, expected_layers: int | None) -> dict[str, Any]:
    keys, weight_map = checkpoint_keys_from_index(model_path)
    source = "model.safetensors.index.json"
    if not keys:
        keys = checkpoint_keys_from_safetensors(model_path)
        source = "safetensors headers"

    if not keys:
        fail("没有读到 checkpoint key，请检查模型目录是否完整")
        return {"keys": set(), "gate_scales": []}

    print(f"从 {source} 读取到 {len(keys)} 个权重 key")
    suffixes = {
        "gate_proj.scale": ".mlp.gate_proj.weight_scale_inv",
        "up_proj.scale": ".mlp.up_proj.weight_scale_inv",
        "down_proj.scale": ".mlp.down_proj.weight_scale_inv",
    }
    matches: dict[str, list[str]] = {}
    for label, suffix in suffixes.items():
        matches[label] = sorted(key for key in keys if key.endswith(suffix))
        print(f"{label}: {len(matches[label])} 个")

    gate_scales = matches["gate_proj.scale"]
    if expected_layers is not None and len(gate_scales) == expected_layers:
        ok(f"checkpoint 中 gate_proj scale 数量与层数一致: {expected_layers}")
    elif gate_scales:
        warn(f"checkpoint gate_proj scale={len(gate_scales)}, expected layers={expected_layers}")
    else:
        fail("checkpoint 中没有 gate_proj.weight_scale_inv")

    for key in gate_scales[:2] + gate_scales[-2:]:
        shard = weight_map.get(key)
        shard_info = f" -> {shard}" if shard else ""
        print(f"  {key}{shard_info}")
        if shard and not (model_path / shard).is_file():
            fail(f"缺失分片文件: {shard}")

    return {"keys": keys, "gate_scales": gate_scales, "matches": matches}


def print_environment(torch: Any, transformers: Any) -> None:
    section("环境")
    print("Python:", sys.version.replace("\n", " "))
    print("平台:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("PyTorch compile CUDA:", torch.version.cuda)
    print("Transformers:", transformers.__version__)
    print("Accelerate:", package_version("accelerate"))
    print("Safetensors:", package_version("safetensors"))
    print("kernels:", package_version("kernels"))
    print("kernels-data:", package_version("kernels-data"))
    print("flash-linear-attention:", package_version("flash-linear-attention"))
    print("causal-conv1d:", package_version("causal-conv1d"))
    print("nvcc path:", shutil.which("nvcc") or "未找到")
    if shutil.which("nvcc"):
        print(run_text_command(["nvcc", "--version"]))

    if not torch.cuda.is_available():
        fail("CUDA 不可用，无法运行 FP8 模型")
        return

    print("GPU 数量:", torch.cuda.device_count())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_gib = props.total_memory / 1024**3
        print(
            f"GPU {index}: {props.name}, {total_gib:.2f} GiB, "
            f"compute capability={props.major}.{props.minor}"
        )
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) >= (8, 9):
        ok("GPU compute capability 满足 Transformers FineGrained FP8 >= 8.9 要求")
    else:
        fail(f"GPU compute capability={major}.{minor}，不满足 FP8 >= 8.9 要求")

    try:
        from transformers.integrations import hub_kernels

        mapping = hub_kernels._HUB_KERNEL_MAPPING.get("finegrained-fp8")
        print("Transformers finegrained-fp8 kernel mapping:", mapping)
    except Exception as exc:
        warn(f"无法读取 FP8 kernel mapping: {exc}")

    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        print("causal-conv1d available:", is_causal_conv1d_available())
        print("flash-linear-attention available:", is_flash_linear_attention_available())
        if not (is_causal_conv1d_available() and is_flash_linear_attention_available()):
            warn("线性注意力将回退到 PyTorch；这会变慢，但不应产生随机多语种乱码")
    except Exception as exc:
        warn(f"无法检查线性注意力插件: {exc}")


def input_device(model: Any, requested_device: str) -> Any:
    import torch

    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return torch.device(requested_device)


def inspect_loaded_gate_modules(model: Any, expected_layers: int | None) -> dict[str, Any]:
    import torch

    section("加载后 gate_proj 完整性")
    modules = [(name, module) for name, module in model.named_modules() if name.endswith(".mlp.gate_proj")]
    print("gate_proj 模块数:", len(modules))
    if expected_layers is not None and len(modules) != expected_layers:
        warn(f"gate_proj 模块={len(modules)}, expected layers={expected_layers}")

    type_counts: Counter[str] = Counter()
    weight_dtype_counts: Counter[str] = Counter()
    missing_scale: list[str] = []
    bad_scale: list[str] = []
    bad_weight_dtype: list[str] = []

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    for name, module in modules:
        type_counts[type(module).__name__] += 1
        weight = getattr(module, "weight", None)
        weight_dtype = getattr(weight, "dtype", None)
        weight_dtype_counts[str(weight_dtype)] += 1
        if fp8_dtype is not None and weight_dtype != fp8_dtype:
            bad_weight_dtype.append(name)

        scale = getattr(module, "weight_scale_inv", None)
        if not isinstance(scale, torch.Tensor):
            missing_scale.append(name)
            continue
        scale_f32 = scale.detach().float()
        finite = bool(torch.isfinite(scale_f32).all().item())
        positive = bool((scale_f32 > 0).all().item())
        if not finite or not positive:
            bad_scale.append(name)

    print("模块类型分布:", dict(type_counts))
    print("权重 dtype 分布:", dict(weight_dtype_counts))
    print("缺失 weight_scale_inv:", len(missing_scale))
    print("非有限或非正 scale:", len(bad_scale))
    print("非 float8_e4m3fn gate 权重:", len(bad_weight_dtype))

    for name, module in (modules[:1] + modules[-1:]):
        scale = getattr(module, "weight_scale_inv", None)
        scale_desc = "缺失"
        if isinstance(scale, torch.Tensor):
            scale_f32 = scale.detach().float()
            scale_desc = (
                f"shape={tuple(scale.shape)}, dtype={scale.dtype}, "
                f"min={scale_f32.min().item():.6g}, max={scale_f32.max().item():.6g}"
            )
        print(
            f"  {name}: type={type(module).__name__}, "
            f"weight={getattr(getattr(module, 'weight', None), 'dtype', None)}, scale={scale_desc}"
        )

    if missing_scale:
        fail(
            "gate_proj 没有 weight_scale_inv。这会导致 FP8 gate 权重被错误解释，"
            "与当前乱码现象高度一致。"
        )
        for name in missing_scale[:5]:
            print("  missing:", name)
    elif bad_scale:
        fail("gate_proj scale 包含非有限数或非正值")
    elif bad_weight_dtype:
        fail("gate_proj 没有按 FP8Linear/float8_e4m3fn 形式加载")
    else:
        ok("所有 gate_proj 都有 FP8 权重和有限正数 weight_scale_inv")

    return {
        "modules": modules,
        "missing_scale": missing_scale,
        "bad_scale": bad_scale,
        "bad_weight_dtype": bad_weight_dtype,
    }


def inspect_loading_info(loading_info: dict[str, Any]) -> dict[str, list[str]]:
    section("权重加载报告")
    result: dict[str, list[str]] = {}
    for category in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        values = list(loading_info.get(category, []) or [])
        result[category] = [str(value) for value in values]
        print(f"{category}: {len(values)}")
        for value in values[:8]:
            print(" ", value)

    bad_unexpected = [
        key for key in result["unexpected_keys"] if ".mlp.gate_proj.weight_scale_inv" in key
    ]
    if bad_unexpected:
        fail(f"仍有 {len(bad_unexpected)} 个 gate_proj scale 被丢弃")
    else:
        ok("加载报告中没有被丢弃的 gate_proj scale")
    return result


def run_generation_test(
    model: Any,
    processor: Any,
    prompt: str,
    requested_device: str,
    enable_thinking: bool,
    max_new_tokens: int,
) -> str:
    import torch

    section("确定性生成测试")
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = input_device(model, requested_device)
    inputs = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }
    input_length = int(inputs["input_ids"].shape[1])
    decoded_prompt = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    print("input_ids shape:", tuple(inputs["input_ids"].shape))
    print("解码后 prompt（这个结果用来判断输入是否真乱码）:")
    print(decoded_prompt)
    print("prompt repr:", repr(decoded_prompt))

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        # 先单独检查 prefill logits，可以快速发现 NaN/Inf。
        prefill = model(**inputs, use_cache=True, return_dict=True)
        next_logits = prefill.logits[:, -1, :].float()
        finite = bool(torch.isfinite(next_logits).all().item())
        print("prefill logits shape:", tuple(prefill.logits.shape))
        print("prefill logits all finite:", finite)
        if finite:
            top_values, top_ids = torch.topk(next_logits[0], k=10)
            print("首 token Top-10:")
            for token_id, logit in zip(top_ids.tolist(), top_values.tolist()):
                token_text = processor.tokenizer.decode([token_id], skip_special_tokens=False)
                print(f"  id={token_id:<7} logit={logit:>11.5f} token={token_text!r}")
        else:
            fail("prefill logits 含 NaN/Inf，数值路径已损坏")
        del prefill, next_logits

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    new_ids = generated_ids[:, input_length:]
    raw_text = processor.tokenizer.decode(new_ids[0], skip_special_tokens=False)
    clean_text = processor.tokenizer.decode(new_ids[0], skip_special_tokens=True)
    token_ids = new_ids[0].tolist()
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    print("生成 token ids:", token_ids)
    print("生成内容（保留 special tokens）:")
    print(raw_text)
    print("生成内容（过滤 special tokens）:")
    print(clean_text)
    print("生成内容 repr:", repr(clean_text))
    print(f"峰值显存: {peak_gib:.2f} GiB")

    if "\ufffd" in clean_text:
        warn("输出含 Unicode replacement character（�）")
    if len(token_ids) >= 8:
        most_common_id, count = Counter(token_ids).most_common(1)[0]
        if count >= max(8, len(token_ids) // 2):
            repeated = processor.tokenizer.decode([most_common_id], skip_special_tokens=False)
            warn(f"检测到异常 token 重复: id={most_common_id}, token={repeated!r}, count={count}")

    return clean_text


def print_conclusion(
    checkpoint_info: dict[str, Any],
    gate_info: dict[str, Any],
    loading_info: dict[str, list[str]],
    generated_text: str,
) -> None:
    section("自动结论")
    gate_unexpected = [
        key
        for key in loading_info.get("unexpected_keys", [])
        if ".mlp.gate_proj.weight_scale_inv" in key
    ]
    has_checkpoint_scales = bool(checkpoint_info.get("gate_scales"))
    loaded_scales_ok = not gate_info.get("missing_scale") and not gate_info.get("bad_scale")
    dtype_ok = not gate_info.get("bad_weight_dtype")

    if has_checkpoint_scales and (gate_unexpected or not loaded_scales_ok or not dtype_ok):
        fail(
            "checkpoint 里有 gate_proj FP8 scale，但模型没有正确接收它们。"
            "这就是当前随机多语种输出的最可能根因。"
        )
    elif has_checkpoint_scales and loaded_scales_ok and dtype_ok and not gate_unexpected:
        ok("FP8 gate_proj 权重与 scale 加载链路正常")
        if generated_text.strip():
            print("请人工检查上面生成的句子是否语义正常。")
    else:
        warn("信息不足，请保留完整输出进一步分析")

    print(
        "\n说明：convert_ids_to_tokens() 显示的 'è¯·...' 是 byte-level BPE 的可视化形式，"
        "不能单独证明 UTF-8 乱码；应以 tokenizer.decode() 和最终语义为准。"
    )
    print(
        "说明：flash-linear-attention/causal-conv1d 缺失时会回退到 PyTorch，"
        "它们影响性能和显存，不应导致随机多语种 token。"
    )


def main() -> int:
    configure_utf8_output()
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()

    if not model_path.is_dir():
        fail(f"模型目录不存在: {model_path}")
        return 2

    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    except Exception:
        fail("导入 PyTorch/Transformers 失败")
        traceback.print_exc()
        return 2

    print_environment(torch, transformers)
    if not torch.cuda.is_available() and not args.inspect_only:
        return 2

    section("Config 与可疑匹配规则")
    try:
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    except Exception:
        fail("AutoConfig 加载失败")
        traceback.print_exc()
        return 2

    print("config class:", type(config).__name__)
    print("model_type:", getattr(config, "model_type", None))
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    expected_layers = getattr(text_config, "num_hidden_layers", None)
    print("text hidden layers:", expected_layers)

    inspect_and_patch_gate_patterns(config, apply_fix=not args.no_fix_gate_pattern)

    section("Checkpoint FP8 scale key")
    checkpoint_info = inspect_checkpoint(model_path, expected_layers)

    if args.inspect_only:
        section("结束")
        print("--inspect-only 已完成，未加载模型。")
        return 0

    section("加载 Processor")
    try:
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        ok(f"processor={type(processor).__name__}")
    except Exception:
        fail("Processor 加载失败")
        traceback.print_exc()
        return 2

    section("加载 FP8 模型")
    try:
        loaded = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(model_path),
            config=config,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": args.device},
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            output_loading_info=True,
        )
        if isinstance(loaded, tuple) and len(loaded) == 2:
            model, raw_loading_info = loaded
        else:
            # 理论上 output_loading_info=True 应返回 tuple，但为未来 API 变化留兼容。
            model, raw_loading_info = loaded, {}
        model.eval()
        ok(f"model={type(model).__name__}, device={model.device}")
    except ImportError as exc:
        fail(f"模型加载缺少依赖: {exc}")
        if "finegrained-fp8" in str(exc) or "kernels" in str(exc):
            print("请先执行: python -m pip install --no-cache-dir kernels==0.15.2")
        return 2
    except Exception:
        fail("模型加载失败")
        traceback.print_exc()
        return 2

    loading_info = inspect_loading_info(raw_loading_info)
    gate_info = inspect_loaded_gate_modules(model, expected_layers)

    try:
        generated_text = run_generation_test(
            model=model,
            processor=processor,
            prompt=args.prompt,
            requested_device=args.device,
            enable_thinking=args.enable_thinking,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception:
        fail("最小生成测试失败")
        traceback.print_exc()
        return 2

    print_conclusion(checkpoint_info, gate_info, loading_info, generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
