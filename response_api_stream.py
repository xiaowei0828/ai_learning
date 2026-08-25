#!/usr/bin/env python3
"""使用 vLLM 的 OpenAI-compatible Responses API 进行流式输出。"""

import argparse
import os

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 vLLM Responses API 并流式打印结果")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="解释一下什么是 KV Cache",
        help="发送给模型的问题",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="vLLM OpenAI API 地址",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VLLM_API_KEY", "EMPTY"),
        help="vLLM API Key",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VLLM_MODEL", "qwen3.8-27b-fp8"),
        help="vLLM 对外暴露的模型名称",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    stream = client.responses.create(
        model=args.model,
        input=args.prompt,
        store=False,
        stream=True,
    )

    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
