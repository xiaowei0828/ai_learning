from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)

response = client.responses.create(
    model="qwen3.8-27b-fp8",
    input="解释一下什么是 KV Cache",
    store=False,
)

print(response.output_text)