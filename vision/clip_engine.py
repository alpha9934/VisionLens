# vision/clip_engine.py

import torch
import open_clip
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[CLIP] Loading model on {device}...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)
model.eval()
tokenizer = open_clip.get_tokenizer("ViT-B-32")

print("[CLIP] Model ready ✅")


def embed_image_from_url(url: str) -> list[float] | None:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        img = Image.open(BytesIO(response.content)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(tensor)
            norm = features / features.norm(dim=-1, keepdim=True)
            return norm.tolist()[0]
    except Exception as e:
        print(f"  [CLIP] URL error: {e}")
        return None


def embed_text(text: str) -> list[float]:
    try:
        tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            norm = features / features.norm(dim=-1, keepdim=True)
            return norm.tolist()[0]
    except Exception as e:
        print(f"  [CLIP] Text error: {e}")
        return [0.0] * 512


def embed_image_from_file(image_bytes: bytes) -> list[float] | None:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(tensor)
            norm = features / features.norm(dim=-1, keepdim=True)
            return norm.tolist()[0]
    except Exception as e:
        print(f"  [CLIP] File error: {e}")
        return None


if __name__ == "__main__":
    print("\n[TEST 1] Text embedding...")
    vec = embed_text("grey sedan with minor dents on front bumper")
    print(f"  Vector dim: {len(vec)}")
    print(f"  First 5 values: {[round(v, 4) for v in vec[:5]]}")

    print("\n[TEST 2] Image from URL...")
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg"
    img_vec = embed_image_from_url(url)
    if img_vec:
        print(f"  Vector dim: {len(img_vec)}")
        print(f"  First 5 values: {[round(v, 4) for v in img_vec[:5]]}")

    print("\n✅ CLIP engine ready!")