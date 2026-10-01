import os

import config  # loads .env (HTW_API_KEY) into the environment

from openai import OpenAI

# ============================================================
# SETTINGS
# ============================================================

API_KEY = os.environ.get("HTW_API_KEY")  # never commit the key: export HTW_API_KEY=...
if not API_KEY:
    raise SystemExit("Set the environment variable HTW_API_KEY (see .env.example).")

BASE_URL = os.environ.get("HTW_BASE_URL", "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1")

MODEL = "qwen3.8-27b"


# ============================================================
# CONNECT TO HTW API
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================
# CHAT LOOP
# ============================================================

# Holds the full conversation so the model keeps the context
# of earlier messages. Type "exit" or "quit" to end the chat.
messages = []

print("Chat gestartet. Zum Beenden 'exit' oder 'quit' eingeben.\n")

while True:
    prompt = input("Du: ").strip()

    if prompt.lower() in ("exit", "quit"):
        print("Chat beendet.")
        break

    if not prompt:
        continue

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})

    print(f"\nKI: {answer}\n")
