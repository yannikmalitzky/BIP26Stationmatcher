import os

import config  # loads .env (HTW_API_KEY) into the environment

from openai import OpenAI
import time

API_KEY = os.environ.get("HTW_API_KEY")  # never commit the key: export HTW_API_KEY=...
if not API_KEY:
    raise SystemExit("Set the environment variable HTW_API_KEY (see .env.example).")

BASE_URL = os.environ.get("HTW_BASE_URL", "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


def main():
    print("=" * 60)
    print("HTW Ollama API TEST")
    print("=" * 60)
    print(f"API: {BASE_URL}")

    # ---------------------------------------------------------
    # 1. Test connection and list available models
    # ---------------------------------------------------------
    print("\n[1] Testing connection / listing models...")

    try:
        models = client.models.list()

        print("Connection successful!")
        print("\nAvailable models:")

        for model in models.data:
            print(f"  - {model.id}")

    except Exception as e:
        print("\nConnection failed!")
        print(f"Error: {e}")
        return

    # ---------------------------------------------------------
    # 2. Select a model
    # ---------------------------------------------------------
    if not models.data:
        print("\nNo models were returned by the server.")
        return

    model = models.data[0].id
    print(f"\nUsing model: {model}")

    # ---------------------------------------------------------
    # 3. Test a simple chat completion
    # ---------------------------------------------------------
    print("\n[2] Testing chat completion...")
    print("-" * 60)

    start = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for HTW Berlin students."
                },
                {
                    "role": "user",
                    "content": "Explain what generative AI is in three short sentences."
                },
            ],
            temperature=0.2,
        )

        elapsed = time.time() - start

        answer = response.choices[0].message.content

        print(answer)
        print("-" * 60)
        print(f"Response time: {elapsed:.2f} seconds")

        if response.usage:
            print(f"Input tokens:  {response.usage.prompt_tokens}")
            print(f"Output tokens: {response.usage.completion_tokens}")
            print(f"Total tokens:  {response.usage.total_tokens}")

    except Exception as e:
        print(f"Chat completion failed: {e}")
        return

    # ---------------------------------------------------------
    # 4. Test streaming
    # ---------------------------------------------------------
    print("\n[3] Testing streaming...")
    print("-" * 60)

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Write a short two-sentence description of Berlin."
                }
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n" + "-" * 60)
        print("Streaming test successful.")

    except Exception as e:
        print(f"Streaming test failed: {e}")

    print("\n" + "=" * 60)
    print("API TEST FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
