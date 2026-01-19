import json
import os
import hashlib
from io import BytesIO
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image


BUILD_MARKER = "INFERENCE_DUALEYE_RESNET50_MATCH_TRAINING_2026_01_19"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")

DEVICE = "cpu"


# -----------------------------
# Labels
# Your labels.json format:
# [
#  {"code":"N","name":"normal"}, ...
# ]
# -----------------------------
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

if isinstance(labels_raw, list) and labels_raw and isinstance(labels_raw[0], dict):
    LABEL_CODES = [x["code"] for x in labels_raw]
    CODE_TO_NAME_RAW = {x["code"]: x["name"] for x in labels_raw}
else:
    # fallback
    LABEL_CODES = list(labels_raw)
    CODE_TO_NAME_RAW = {
        "N": "normal",
        "D": "diabetic_retinopathy",
        "G": "glaucoma",
        "C": "cataract",
        "A": "amd",
        "H": "hypertensive_retinopathy",
        "M": "myopia",
        "O": "other",
    }

NUM_CLASSES = len(LABEL_CODES)


def _ui_name(name: str) -> str:
    if not name:
        return "Uncertain"
    name = name.replace("_", " ").replace("-", " ").strip()
    return " ".join(w.capitalize() for w in name.split())


def translate_code(code: str) -> str:
    if not code:
        return "Uncertain"
    return _ui_name(CODE_TO_NAME_RAW.get(code, code))


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Model (MATCH TRAINING)
# -----------------------------
class DualEyeModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, left, right):
        left_feat = self.backbone(left)
        right_feat = self.backbone(right)
        combined = torch.cat((left_feat, right_feat), dim=1)
        return self.classifier(combined)


# -----------------------------
# Preprocess (MATCH TRAINING)
# Training used only Resize + ToTensor
# (no ImageNet normalize!)
# -----------------------------
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


def load_model() -> Tuple[nn.Module, str, str, int]:
    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(f"Model weights not found: {WEIGHTS_PATH}")

    model = DualEyeModel(NUM_CLASSES).to(DEVICE)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    missing, unexpected = model.load_state_dict(state, strict=False)

    # We accept strict=False because sometimes training adds prefixes,
    # BUT missing/unexpected should normally be empty for exact match.
    load_report = f"missing={len(missing)} unexpected={len(unexpected)}"

    model.eval()
    return model, load_report, _sha256_file(WEIGHTS_PATH), os.path.getsize(WEIGHTS_PATH)


MODEL, LOAD_REPORT, WEIGHTS_SHA, WEIGHTS_SIZE = load_model()
ACTIVE_VARIANT = "dualeye_resnet50"


def _tensor_from_bytes(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)  # [1,3,224,224]
    return x


def predict_raw(left_bytes: bytes, right_bytes: bytes) -> dict:
    left_x = _tensor_from_bytes(left_bytes)
    right_x = _tensor_from_bytes(right_bytes)

    with torch.no_grad():
        out = MODEL(left_x, right_x)[0]  # [8]
        probs = out.detach().cpu().tolist()

    prob_map = {LABEL_CODES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    top_code = max(prob_map, key=prob_map.get)
    top_prob = prob_map[top_code]
    top3 = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)[:3]

    return {
        "top_code": top_code,
        "top_prob": float(top_prob),
        "top3": [(k, float(v)) for k, v in top3],
        "probs": prob_map,
    }


def predict_diagnosis(left_bytes: bytes, right_bytes: bytes, threshold: float = 0.50) -> str:
    raw = predict_raw(left_bytes, right_bytes)
    if raw["top_prob"] < threshold:
        return "Uncertain"
    return translate_code(raw["top_code"])


def predict_debug(left_bytes: bytes, right_bytes: bytes) -> dict:
    raw = predict_raw(left_bytes, right_bytes)
    return {
        "build_marker": BUILD_MARKER,
        "active_variant": ACTIVE_VARIANT,
        "weights_sha": WEIGHTS_SHA,
        "weights_size": WEIGHTS_SIZE,
        "load_report": LOAD_REPORT,
        "labels": LABEL_CODES,
        **raw,
        "translated": translate_code(raw["top_code"]),
    }
