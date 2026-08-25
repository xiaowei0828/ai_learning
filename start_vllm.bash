export VLLM_ENABLE_RESPONSES_API_STORE=1

CUDA_VISIBLE_DEVICES=0 vllm serve /root/ai-models/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b-fp8 \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "$VLLM_API_KEY" \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-num-seqs 8