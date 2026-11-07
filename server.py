"""
Deepfake detector backend: your Model B (EfficientNet-B0 + Bi-LSTM) + Sightengine / Hugging Face APIs + fusion.

Run:
    pip install fastapi uvicorn python-multipart requests torch torchvision pillow
    set SIGHTENGINE_USER=your_user          (Windows CMD; use $env:... in PowerShell)
    set SIGHTENGINE_SECRET=your_secret
    set HF_TOKEN=your_token                 (optional, used only if no Sightengine key)
    uvicorn server:app --port 8000
Then open http://localhost:8000 in Chrome.
"""
import io
import os
import re

import requests
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from torchvision import models, transforms

# ---------------- settings ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "modelB_8data_best.pth")
SE_USER = os.getenv("SIGHTENGINE_USER", "")
SE_SECRET = os.getenv("SIGHTENGINE_SECRET", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODELS = ["prithivMLmods/Deep-Fake-Detector-v2-Model", "umm-maybe/AI-image-detector"]

W_DL = float(os.getenv("W_DL", "0.5"))   # weight of your model in the fusion (API gets 1 - W_DL)
THRESHOLD = 0.5                          # final score >= THRESHOLD -> FAKE


# ---------------- your Model B ----------------
class EffNetBiLSTM(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.bilstm = nn.LSTM(input_size=1280, hidden_size=256, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(p), nn.Linear(128, 1))

    def forward(self, x):
        f = self.features(x)
        f = f.flatten(2).permute(0, 2, 1)
        out, _ = self.bilstm(f)
        return self.fc(self.drop(out[:, -1, :])).squeeze(1)


model = EffNetBiLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_model(img: Image.Image) -> float:
    """Returns P(fake) from your trained model."""
    with torch.no_grad():
        return torch.sigmoid(model(preprocess(img).unsqueeze(0))).item()


# ---------------- API detectors ----------------
def sightengine(data: bytes, content_type: str):
    """Returns {'ai': float|None, 'face': float|None}."""
    def call(models_):
        r = requests.post(
            "https://api.sightengine.com/1.0/check.json",
            files={"media": ("image", data, content_type)},
            data={"models": models_, "api_user": SE_USER, "api_secret": SE_SECRET},
            timeout=30,
        )
        return r.json()

    j = call("genai,deepfake")
    if j.get("status") != "success":
        j = call("genai")
    if j.get("status") != "success":
        raise RuntimeError("Sightengine: " + str((j.get("error") or {}).get("message", "request failed")))
    t = j.get("type", {})
    return {"ai": t.get("ai_generated"), "face": t.get("deepfake")}


def fake_prob(labels):
    for x in labels:
        if re.search(r"fake|artificial|\bai\b|generated|synthetic", x["label"], re.I):
            return x["score"]
    for x in labels:
        if re.search(r"real|human|authentic|natural", x["label"], re.I):
            return 1 - x["score"]
    return None


def hugging_face(data: bytes, content_type: str):
    scores = []
    for m in HF_MODELS:
        try:
            r = requests.post(
                "https://router.huggingface.co/hf-inference/models/" + m,
                headers={"Authorization": "Bearer " + HF_TOKEN, "Content-Type": content_type},
                data=data, timeout=40,
            )
            j = r.json()
            if r.ok and isinstance(j, list):
                p = fake_prob(j[0] if j and isinstance(j[0], list) else j)
                if p is not None:
                    scores.append(p)
        except Exception:
            pass
    if not scores:
        raise RuntimeError("Hugging Face: no usable result")
    return sum(scores) / len(scores)


def metadata_check(data: bytes, content_type: str):
    s = data[:400000].decode("latin1")
    hit = None
    if re.search(r"trainedAlgorithmicMedia", s, re.I):
        hit = "says it was made or edited by AI"
    elif re.search(r"openai|chatgpt|gpt-image|dall-?e|midjourney|stable diffusion|firefly|gemini|synthid", s, re.I):
        hit = "names an AI image tool as its source"
    return hit, (content_type == "image/jpeg" and "Exif" not in s)


# ---------------- app ----------------
app = FastAPI()


@app.get("/")
def home():
    return FileResponse("deepfake-detector.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    ctype = file.content_type or "image/jpeg"
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Branch 1: your deep learning model
    dl = run_model(img)

    # Branch 2: API detector
    api, se_ai, se_face, hf, api_error = None, None, None, None, None
    try:
        if SE_USER and SE_SECRET:
            r = sightengine(data, ctype)
            se_ai, se_face = r["ai"], r["face"]
            vals = [v for v in (se_ai, se_face) if isinstance(v, (int, float))]
            api = max(vals) if vals else None
        elif HF_TOKEN:
            hf = hugging_face(data, ctype)
            api = hf
    except Exception as e:
        api_error = str(e)

    meta_hit, no_camera = metadata_check(data, ctype)

    # Decision / fusion
    final = dl if api is None else W_DL * dl + (1 - W_DL) * api
    is_fake = bool(meta_hit) or final >= THRESHOLD

    return {
        "label": "FAKE" if is_fake else "REAL",
        "final_score": final,
        "dl_score": dl,
        "api_score": api,
        "se_ai": se_ai,
        "se_face": se_face,
        "hf": hf,
        "meta_hit": meta_hit,
        "no_camera": no_camera,
        "api_error": api_error,
        "weights": {"dl": W_DL, "api": 1 - W_DL},
    }
