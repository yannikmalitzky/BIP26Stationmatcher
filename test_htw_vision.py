import os

import config  # loads .env (HTW_API_KEY) into the environment

from openai import OpenAI
import base64
import mimetypes

# ============================================================
# SETTINGS
# ============================================================

API_KEY = os.environ.get("HTW_API_KEY")  # never commit the key: export HTW_API_KEY=...
if not API_KEY:
    raise SystemExit("Set the environment variable HTW_API_KEY (see .env.example).")

BASE_URL = os.environ.get("HTW_BASE_URL", "https://f2ki-h100-1.f2.htw-berlin.de:11435/v1")

# Change this to a vision-capable model available on the server
MODEL = "qwen3.8-27b"


# ============================================================
# CONNECT TO HTW API
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================
# LOAD IMAGE
# ============================================================

image_path = input("Enter the path to your image: ").strip()

mime_type, _ = mimetypes.guess_type(image_path)

if mime_type is None:
    mime_type = "image/jpeg"

with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")


# ============================================================
# SEND IMAGE TO MODEL
# ============================================================

print("\nAnalyzing image...\n")

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Search for any Text in this picture and list it. Select the Part which describes the location most likely. Compare it with current and historical U Bahn Stations in Berlin and tell to which of those it belongs"
                        
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    },
                },
            ],
        }
    ],
)

# ============================================================
# PRINT RESULT
# ============================================================

print("=" * 60)
print("IMAGE DESCRIPTION")
print("=" * 60)

print(response.choices[0].message.content)