# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
from google import genai
from google.genai import types

OUTPUT_DIR = '<OUTPUT_DIR>'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = """Generate different variations of <WHAT YOU WANT>:
1. Variation 1 description
2. Variation 2 description
"""

IMAGE_NAME_BASE = '<IMAGE_NAME_BASE>'

def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to to: {file_name}")


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-image"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=PROMPT),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            file_name = f"{IMAGE_NAME_BASE}_{file_index}"
            file_index += 1
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            save_binary_file(os.path.join(OUTPUT_DIR, f"{file_name}{file_extension}"), data_buffer)
        else:
            print(chunk.text)

if __name__ == "__main__":
    generate()