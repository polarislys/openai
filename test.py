from openai import OpenAI

# 连接到你自己的 Rust 服务器，而不是 OpenAI 云端
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed"  # Rust 服务不会验证 API key
)

# 测试非流式请求
resp = client.chat.completions.create(
    model="test",
    messages=[
        {"role": "user", "content": "Hello from Python client!"}
    ]
)

print("非流式响应：")
print(resp)

# 测试流式请求
print("\n流式响应：")
stream = client.chat.completions.create(
    model="test",
    messages=[{"role": "user", "content": "Stream test from Python"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
