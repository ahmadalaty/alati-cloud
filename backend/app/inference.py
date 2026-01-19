import json
import os
import hashlib
from io import BytesIO
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

BUILD_MARKER = "INFERENCE_FINAL_DUALEYE_MATCH_TRAIN_NORMAL_SUPPRESS_2026_01_19"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# labels.json format:
# [
#   {"code":"N","name":"normal"},
#   {"code":"D","name":"diabetic_retinopathy"},
#   ...
# ]
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

if isinstance(labels_raw, list) and len(labels_raw) > 0 and isinstance(labels_raw[0], dict):
    LABELS: List[str] = [x["code"] for x in labels_raw]
    CODE_TO_NAME = {x["code"]: x["name"] for x in labels_raw}
else:
    LABELS = list(labels_raw)
    CODE_TO_NAME = {
        "N": "normal",
        "D": "diabetic_retinopathy",
        "G": "glaucoma",
        "C": "cataract",
        "A": "amd",
        "H": "hypertensive_retinopathy",
        "M": "myopia",
        "O": "other",
    }

NUM_CLASSES = len(LABELS)
DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower()
DEVICE = "cpu"

# Thresholds (tunable)
N_THRESH = float(os.getenv("N_THRESH", "0.70"))                    # Normal strong
DISEASE_BLOCK_THRESH = float(os.getenv("DISEASE_BLOCK", "0.35"))   # Any disease above this blocks Normal
DISEASE_MIN_THRESH = float(os.getenv("DISEASE_MIN", "0.50"))       # Disease must be >= this to output disease
MIRROR_TTA = str(os.getenv("MIRROR_TTA", "1")).strip() == "1"       # mirror augmentation on/off


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _polish_name(name: str) -> str:
    if not name:
        return "Uncertain"
    name = name.replace("_", " ").replace("-", " ").strip()
    name = " ".join(w.capitalize() for w in name.split())
    return name or "Uncertain"


def translate_code(code: str) -> str:
    if not code:
        return "Uncertain"
    return _polish_name(CODE_TO_NAME.get(code, code))


class DualEyeModel(nn.Module):
    """
    MUST MATCH TRAINING EXACTLY.
    """
    def __init__(self, num_classes=8, backbone_name="resnet18"):
        super().__init__()
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
            out_features = 2048
        else:
            self.backbone = models.resnet18(weights=None)
            out_features = 512

        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(out_features * 2, 256),
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


TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        # (training did not normalize, so DON'T add normalization!)
    ]
)


def load_model(model_variant: str):
    model_variant = (model_variant or "resnet18").strip().lower()
    if model_variant == "resnet50":
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
        active = "resnet50"
    else:
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
        active = "resnet18"

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")

    weights_size = os.path.getsize(weights_path)
    weights_sha = _sha256_bytes(open(weights_path, "rb").read())

    model = DualEyeModel(num_classes=NUM_CLASSES, backbone_name=active).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)

    # strict True because architecture now matches training
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, active, "strict", weights_sha, weights_size


MODEL, ACTIVE_VARIANT, LOAD_MODE, WEIGHTS_SHA, WEIGHTS_SIZE = load_model(DEFAULT_VARIANT)


def _tensor_from_bytes(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)  # [1,3,224,224]
    return x


def _avg_probs(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        out[k] = float((a.get(k, 0.0) + b.get(k, 0.0)) / 2.0)
    return out


def _probs_from_bytes_single(image_bytes: bytes) -> Dict[str, float]:
    """
    Single-eye inference by duplicating the same image as left and right.
    This matches training format (DualEyeModel expects 2 inputs).
    """
    x = _tensor_from_bytes(image_bytes)

    with torch.no_grad():
        probs = MODEL(x, x)[0].detach().cpu().tolist()  # already sigmoid in model

    out = {}
    for i in range(min(len(probs), len(LABELS))):
        out[LABELS[i]] = float(probs[i])
    return out


def _mirror_bytes(image_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def probs_from_bytes(image_bytes: bytes) -> Dict[str, float]:
    """
    Returns probabilities with optional Mirror TTA.
    """
    p1 = _probs_from_bytes_single(image_bytes)

    if not MIRROR_TTA:
        return p1

    flipped = _mirror_bytes(image_bytes)
    p2 = _probs_from_bytes_single(flipped)
    return _avg_probs(p1, p2)


def choose_final_code(probs: Dict[str, float]) -> Tuple[str, str]:
    """
    Multi-label rule:
    - Normal only if strong and no disease is significant.
    - Otherwise take best disease if confident.
    """
    n_prob = float(probs.get("N", 0.0))

    disease_codes = [c for c in LABELS if c != "N"]
    if not disease_codes:
        return "UNCERTAIN", "No disease labels loaded"

    best_disease = max(disease_codes, key=lambda c: probs.get(c, 0.0))
    disease_max = float(probs.get(best_disease, 0.0))

    # ✅ Normal allowed only if all diseases low
    if n_prob >= N_THRESH and disease_max < DISEASE_BLOCK_THRESH:
        return "N", f"Normal allowed: N={n_prob:.3f} disease_max={disease_max:.3f}"

    # ✅ Disease chosen if high enough
    if disease_max >= DISEASE_MIN_THRESH:
        return best_disease, f"Disease chosen: {best_disease}={disease_max:.3f} N={n_prob:.3f}"

    return "UNCERTAIN", f"Uncertain: N={n_prob:.3f} disease_max={disease_max:.3f}"


def predict_raw(image_bytes: bytes) -> dict:
    probs = probs_from_bytes(image_bytes)

    if probs:
        top_code = max(probs, key=probs.get)
        top_prob = float(probs[top_code])
    else:
        top_code, top_prob = None, None

    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    final_code, reason = choose_final_code(probs)
    translated = translate_code(final_code) if final_code != "UNCERTAIN" else "Uncertain"

    return {
        "top_code": top_code,
        "top_prob": top_prob,
        "top3": [(k, float(v)) for k, v in top3],
        "final_code": final_code,
        "final_reason": reason,
        "translated": translated,
        "probs": probs,  # keep for debugging only
    }


def predict_diagnosis(image_bytes: bytes) -> str:
    raw = predict_raw(image_bytes)
    return raw["translated"]


def predict_debug(image_bytes: bytes) -> dict:
    raw = predict_raw(image_bytes)
    return {
        "build_marker": BUILD_MARKER,
        "active_variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "weights_sha": WEIGHTS_SHA,
        "weights_size": WEIGHTS_SIZE,
        "labels": LABELS,
        "num_classes": NUM_CLASSES,
        "mirror_tta": MIRROR_TTA,
        "thresholds": {
            "N_THRESH": N_THRESH,
            "DISEASE_BLOCK_THRESH": DISEASE_BLOCK_THRESH,
            "DISEASE_MIN_THRESH": DISEASE_MIN_THRESH,
        },
        **raw,
    }
