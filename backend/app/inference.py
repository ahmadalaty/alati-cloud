import json
import os
import hashlib
from io import BytesIO
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

BUILD_MARKER = "INFERENCE_DR_FOCUS_2026_01_24"

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# ============ PHASE 1: DIABETIC RETINOPATHY FOCUS ============
# Later phases can add: GLAUCOMA, CATARACT, AMD, etc.

# labels.json format:
# [
#   {"code":"N","name":"normal"},
#   {"code":"D","name":"diabetic_retinopathy"},
#   ...
# ]
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels_raw = json.load(f)

if isinstance(labels_raw, list) and len(labels_raw) > 0 and isinstance(labels_raw[0], dict):
    ALL_LABELS: List[str] = [x["code"] for x in labels_raw]
    CODE_TO_NAME = {x["code"]: x["name"] for x in labels_raw}
else:
    ALL_LABELS = list(labels_raw)
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

# ============ PHASE 1 CONFIG: Focus on DR vs Normal ============
ACTIVE_PHASE = int(os.getenv("ACTIVE_PHASE", "1"))  # 1 = DR only, 2+ = multi-disease (future)

if ACTIVE_PHASE == 1:
    # PHASE 1: Diabetic Retinopathy vs Normal
    ACTIVE_LABELS = ["N", "D"]  # Only Normal and Diabetic Retinopathy
    PHASE_NAME = "Diabetic Retinopathy Detection"
else:
    # PHASE 2+: Multi-disease (future)
    # ACTIVE_LABELS = ["N", "D", "G", "C", "A"]  # Uncomment when ready
    ACTIVE_LABELS = ["N", "D"]  # Default back to Phase 1

LABELS = ACTIVE_LABELS
NUM_CLASSES = len(ALL_LABELS)  # Model still trained on all classes
DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower()
DEVICE = "cpu"

# ============ THRESHOLDS (Tunable per phase) ============
N_THRESH = float(os.getenv("N_THRESH", "0.70"))                    # Normal strong
DISEASE_BLOCK_THRESH = float(os.getenv("DISEASE_BLOCK", "0.35"))   # Any disease above this blocks Normal
DISEASE_MIN_THRESH = float(os.getenv("DISEASE_MIN", "0.50"))       # Disease must be >= this to output disease
MIRROR_TTA = str(os.getenv("MIRROR_TTA", "1")).strip() == "1"       # mirror augmentation on/off

# ============ MODEL VERSION ============
# v1 is the original ODIR dual-eye resnet18 at 224px. v6 is a single-eye
# resnet50 at 768px with cumulative-ordinal heads, trained only on CC BY 4.0
# data (Paraguay + RFMiD 2.0).
#
# Measured on 6,862 images external to both models:
#
#                     APTOS (3,662)        RFMiD 1.0 (3,200)
#   v1   AUC 0.881   75.4% / 88.1%        87.2% / 72.4%
#   v6   AUC 0.967   85.9% / 95.7%        93.0% / 74.8%
#
# v6 is better on both axes on both sets. Set MODEL_VERSION=v1 to roll back
# instantly without a redeploy.
MODEL_VERSION = os.getenv("MODEL_VERSION", "v6").strip().lower()

# v6's calibration does not transfer as well as its ranking does. Its training
# mix is ~35% diseased and skewed to severe disease, so a threshold picked on
# its own held-out split (0.294, which scored 99.1% specificity there) collapses
# to 41-53% specificity on real distributions. 0.898 was also chosen on that
# held-out split - never on the sets quoted above - and is the point at which v6
# beats v1 on both axes on both. Treat it as calibrated for screening
# populations, and re-check it against local data before trusting it on a
# population unlike either.
V6_THRESH = float(os.getenv("V6_THRESH", "0.898"))
V6_WEIGHTS = os.path.join(MODEL_DIR, "alati_dr_v6_ccby.pth")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRADE_LABEL = {0: "No diabetic retinopathy", 1: "Mild", 2: "Moderate",
               3: "Severe", 4: "Proliferative"}

# Data attribution, required by CC BY 4.0 and surfaced through predict_debug:
#   Benitez et al., "Dataset from fundus images for the study of diabetic
#     retinopathy", Hospital de Clinicas, Universidad Nacional de Asuncion,
#     Paraguay. CC BY 4.0. doi:10.5281/zenodo.4891308
#   Panchal, Naik, Kokare, Pachade et al., "Retinal Fundus Multi-disease Image
#     Dataset (RFMiD) 2.0". CC BY 4.0. doi:10.5281/zenodo.7505822
V6_ATTRIBUTION = [
    "Paraguay fundus dataset (Benitez et al.), CC BY 4.0, doi:10.5281/zenodo.4891308",
    "RFMiD 2.0 (Panchal, Naik, Kokare, Pachade et al.), CC BY 4.0, doi:10.5281/zenodo.7505822",
]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ============ IMAGE ENHANCEMENT (Retinal-specific preprocessing) ============

def enhance_retinal_image(image_bytes: bytes) -> bytes:
    """
    Retinal-image enhancement pipeline:
      1. Crop black borders (find non-black bounding box of the retinal disc)
      2. Auto-contrast (stretches histogram - helps under/overexposed images)
      3. Unsharp mask (sharpens vessels and microaneurysms)
      4. Mild contrast boost
    
    Returns enhanced image as JPEG bytes. If anything fails, returns the
    original bytes unchanged (fail-safe).
    """
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # 1. Auto-crop black borders
        arr = np.array(img)
        gray = arr.mean(axis=2)
        mask = gray > 15  # threshold for "not black"
        if mask.any():
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            r_idx = np.where(rows)[0]
            c_idx = np.where(cols)[0]
            if len(r_idx) > 0 and len(c_idx) > 0:
                rmin, rmax = r_idx[0], r_idx[-1]
                cmin, cmax = c_idx[0], c_idx[-1]
                # Only crop if it removes meaningful border (>3% of image)
                h, w = arr.shape[:2]
                if (cmin > w * 0.03 or w - cmax > w * 0.03 or
                    rmin > h * 0.03 or h - rmax > h * 0.03):
                    img = img.crop((cmin, rmin, cmax + 1, rmax + 1))

        # 2. Auto-contrast (histogram stretching) — cutoff=1 ignores top/bottom 1% outliers
        img = ImageOps.autocontrast(img, cutoff=1)

        # 3. Unsharp mask — enhances fine detail (vessels, lesions, microaneurysms)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))

        # 4. Mild contrast boost (~15%)
        img = ImageEnhance.Contrast(img).enhance(1.15)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        # Fail-safe: return original if anything goes wrong
        return image_bytes


# ============ END IMAGE ENHANCEMENT ============


def _polish_name(name: str) -> str:
    if not name:
        return "Other"
    name = name.replace("_", " ").replace("-", " ").strip()
    name = " ".join(w.capitalize() for w in name.split())
    return name or "Other"


def translate_code(code: str) -> str:
    if not code:
        return "Other"
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


# ============ v6: single-eye, 768px, cumulative-ordinal ============

class DRNetV6(nn.Module):
    """
    Single-eye backbone with a cumulative-ordinal head.

    Two departures from DualEyeModel that matter. It takes ONE image, because
    the endpoint serves one - the old model was trained on genuine left/right
    pairs and then called as MODEL(x, x) with one photograph duplicated, so it
    was asked at inference for a relationship it never saw. And the head
    predicts P(grade>=1..4) rather than a flat class, because grades are
    ordered: confusing grade 4 with 3 should not cost what confusing it with 0
    does. P(grade>=1) is "any DR" and P(grade>=2) is referable disease.
    """
    def __init__(self, backbone_name="resnet50"):
        super().__init__()
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
            feat = 2048
        else:
            self.backbone = models.resnet18(weights=None)
            feat = 512
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, 4))

    def forward(self, x):
        return self.head(self.backbone(x))


def _load_v6():
    if not os.path.exists(V6_WEIGHTS):
        raise RuntimeError(f"v6 weights not found: {V6_WEIGHTS}")
    ck = torch.load(V6_WEIGHTS, map_location=DEVICE)
    m = DRNetV6(ck.get("backbone", "resnet50")).to(DEVICE)
    m.load_state_dict(ck["state"], strict=True)
    m.eval()
    tf = T.Compose([
        T.Resize((ck.get("size", 768),) * 2),
        T.ToTensor(),
        # v1 was trained without normalization; v6 was trained with it, against
        # an ImageNet-pretrained backbone. Getting this wrong silently degrades
        # every prediction rather than raising.
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    sha = _sha256_bytes(open(V6_WEIGHTS, "rb").read())
    return m, tf, ck.get("size", 768), sha, os.path.getsize(V6_WEIGHTS)


V6_MODEL = V6_TRANSFORM = None
V6_SIZE = V6_SHA = V6_BYTES = None
if MODEL_VERSION == "v6":
    V6_MODEL, V6_TRANSFORM, V6_SIZE, V6_SHA, V6_BYTES = _load_v6()


def _crop_to_disc(img: Image.Image) -> Image.Image:
    """Trim the black surround to the retinal disc's bounding box."""
    a = np.asarray(img.convert("RGB"))
    m = a.max(axis=2) > 18
    if not m.any():
        return img
    rows = np.where(np.any(m, axis=1))[0]
    cols = np.where(np.any(m, axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return img
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def _square_pad(img: Image.Image) -> Image.Image:
    """Pad to square on black so the disc is not distorted by the resize."""
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (0, 0, 0))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def _v6_predict(image_bytes: bytes) -> dict:
    """
    Returns the same shape as predict_raw so callers do not change, plus the
    severity fields v1 could never produce.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = _square_pad(_crop_to_disc(img))
    x = V6_TRANSFORM(img).unsqueeze(0)
    batch = torch.cat([x, torch.flip(x, dims=[3])], dim=0) if MIRROR_TTA else x
    with torch.no_grad():
        p = torch.sigmoid(V6_MODEL(batch)).mean(dim=0).tolist()   # P(g>=1..4)

    p_any, p_ref = float(p[0]), float(p[1])
    # Grade is how many ordinal thresholds the image clears. Using the same
    # cut-off throughout keeps the reported grade consistent with the yes/no
    # decision - a "grade 2" that the binary call disagreed with would be
    # incoherent.
    grade = sum(1 for v in p if v >= V6_THRESH)
    is_dr = p_any >= V6_THRESH

    # No uncertain band: the quoted sensitivity and specificity describe a
    # two-way decision at this threshold, and adding a third outcome would mean
    # the deployed behaviour is not the behaviour that was measured.
    code = "D" if is_dr else "N"
    return {
        "phase": ACTIVE_PHASE,
        "phase_name": PHASE_NAME,
        "model_version": "v6",
        "top_code": code,
        "top_prob": p_any,
        "top3": [("D", p_any), ("N", 1.0 - p_any)],
        "final_code": code,
        "final_reason": f"v6 ordinal: P(any DR)={p_any:.3f} vs threshold {V6_THRESH:.3f}",
        "translated": translate_code(code),
        "probs": {"N": 1.0 - p_any, "D": p_any},
        "confidence": p_any if is_dr else 1.0 - p_any,
        "enhanced": False,
        # new with v6 - v1 has no notion of severity at all
        "grade": grade,
        "grade_label": GRADE_LABEL.get(grade, "Unknown"),
        "p_any_dr": p_any,
        "p_referable": p_ref,
        "ordinal": [float(v) for v in p],
    }


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
    Returns ALL class probabilities (model trained on all classes).
    """
    x = _tensor_from_bytes(image_bytes)

    with torch.no_grad():
        probs = MODEL(x, x)[0].detach().cpu().tolist()  # already sigmoid in model

    out = {}
    for i in range(min(len(probs), len(ALL_LABELS))):
        out[ALL_LABELS[i]] = float(probs[i])
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
    Returns ALL class probabilities (we filter later based on ACTIVE_PHASE).
    """
    p1 = _probs_from_bytes_single(image_bytes)

    if not MIRROR_TTA:
        return p1

    flipped = _mirror_bytes(image_bytes)
    p2 = _probs_from_bytes_single(flipped)
    return _avg_probs(p1, p2)


def choose_final_code(probs: Dict[str, float]) -> Tuple[str, str]:
    """
    Multi-label rule - PHASE-AWARE:
    
    PHASE 1 (DR focus):
    - Normal only if strong and DR is low.
    - Otherwise return DR if confident.
    - Return "Uncertain" if neither confident.
    
    PHASE 2+ (multi-disease):
    - Can extend to choose best disease among multiple options.
    """
    n_prob = float(probs.get("N", 0.0))

    if ACTIVE_PHASE == 1:
        # ============ PHASE 1: DR vs Normal ============
        d_prob = float(probs.get("D", 0.0))

        # Normal allowed only if strong AND DR is low
        if n_prob >= N_THRESH and d_prob < DISEASE_BLOCK_THRESH:
            return "N", f"Normal: N={n_prob:.3f} DR={d_prob:.3f}"

        # DR chosen if confident enough
        if d_prob >= DISEASE_MIN_THRESH:
            return "D", f"Diabetic Retinopathy: DR={d_prob:.3f} N={n_prob:.3f}"

        # Otherwise uncertain
        return "UNCERTAIN", f"Uncertain: N={n_prob:.3f} DR={d_prob:.3f}"

    else:
        # ============ PHASE 2+: Multi-disease (future) ============
        # This section will expand as you add more diseases
        disease_codes = [c for c in ACTIVE_LABELS if c != "N"]
        if not disease_codes:
            return "UNCERTAIN", "No disease labels in active phase"

        best_disease = max(disease_codes, key=lambda c: probs.get(c, 0.0))
        disease_max = float(probs.get(best_disease, 0.0))

        # Normal allowed only if all diseases low
        if n_prob >= N_THRESH and disease_max < DISEASE_BLOCK_THRESH:
            return "N", f"Normal: N={n_prob:.3f} disease_max={disease_max:.3f}"

        # Disease chosen if high enough
        if disease_max >= DISEASE_MIN_THRESH:
            return best_disease, f"Disease: {best_disease}={disease_max:.3f} N={n_prob:.3f}"

        return "UNCERTAIN", f"Uncertain: N={n_prob:.3f} disease_max={disease_max:.3f}"


def predict_raw(image_bytes: bytes, enhance: bool = False) -> dict:
    """
    Core prediction function.
    Returns all probabilities but filters final output based on ACTIVE_PHASE.
    
    Args:
        image_bytes: raw image bytes
        enhance: if True, apply retinal enhancement before inference
    """
    if MODEL_VERSION == "v6":
        # v6 was trained on cropped, padded, ImageNet-normalized images. The v1
        # enhancement pipeline (autocontrast, unsharp mask, contrast boost) was
        # never part of that, so applying it would be a distribution shift, not
        # a help. v6 does its own disc crop, which is the part that mattered.
        return _v6_predict(image_bytes)

    # Apply enhancement if requested
    enhanced_applied = False
    if enhance:
        original_size = len(image_bytes)
        image_bytes = enhance_retinal_image(image_bytes)
        enhanced_applied = (len(image_bytes) != original_size or enhance)
    
    probs = probs_from_bytes(image_bytes)

    if probs:
        top_code = max(probs, key=probs.get)
        top_prob = float(probs[top_code])
    else:
        top_code, top_prob = None, None

    top3 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]

    final_code, reason = choose_final_code(probs)
    translated = translate_code(final_code) if final_code != "UNCERTAIN" else "Other"

    # ============ PHASE 1: Return only DR and Normal probabilities ============
    if ACTIVE_PHASE == 1:
        active_probs = {k: v for k, v in probs.items() if k in ACTIVE_LABELS}
    else:
        active_probs = {k: v for k, v in probs.items() if k in ACTIVE_LABELS}

    return {
        "phase": ACTIVE_PHASE,
        "phase_name": PHASE_NAME,
        "top_code": top_code,
        "top_prob": top_prob,
        "top3": [(k, float(v)) for k, v in top3],
        "final_code": final_code,
        "final_reason": reason,
        "translated": translated,
        "probs": active_probs,  # Filtered by phase
        "confidence": float(probs.get(final_code, 0.0)) if final_code != "UNCERTAIN" else 0.0,
        "enhanced": enhanced_applied,
    }


def predict_diagnosis(image_bytes: bytes, enhance: bool = False) -> str:
    """
    Simple API: returns diagnosis string only.
    """
    raw = predict_raw(image_bytes, enhance=enhance)
    return raw["translated"]


def predict_debug(image_bytes: bytes, enhance: bool = False) -> dict:
    """
    Full debug output including all metadata.
    """
    raw = predict_raw(image_bytes, enhance=enhance)
    if MODEL_VERSION == "v6":
        return {
            "build_marker": BUILD_MARKER,
            "model_version": "v6",
            "active_variant": "resnet50-ordinal-768",
            "load_mode": "strict",
            "weights_sha": V6_SHA,
            "weights_size": V6_BYTES,
            "input_size": V6_SIZE,
            "threshold": V6_THRESH,
            "trained_on": ["paraguay", "rfmid2"],
            "data_attribution": V6_ATTRIBUTION,
            "mirror_tta": MIRROR_TTA,
            **raw,
        }
    return {
        "build_marker": BUILD_MARKER,
        "model_version": "v1",
        "active_variant": ACTIVE_VARIANT,
        "load_mode": LOAD_MODE,
        "weights_sha": WEIGHTS_SHA,
        "weights_size": WEIGHTS_SIZE,
        "active_phase": ACTIVE_PHASE,
        "phase_name": PHASE_NAME,
        "active_labels": ACTIVE_LABELS,
        "all_labels": ALL_LABELS,
        "num_classes": NUM_CLASSES,
        "mirror_tta": MIRROR_TTA,
        "thresholds": {
            "N_THRESH": N_THRESH,
            "DISEASE_BLOCK_THRESH": DISEASE_BLOCK_THRESH,
            "DISEASE_MIN_THRESH": DISEASE_MIN_THRESH,
        },
        **raw,
    }