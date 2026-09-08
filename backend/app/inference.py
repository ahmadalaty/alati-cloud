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

# A container's CPU quota is not what torch sees. torch.get_num_threads() reads
# the host's core count, not the cgroup limit, so on a 1-CPU instance it spawns a
# dozen intra-op workers that contend for one core and each carry their own
# buffers - slower and heavier than running single-threaded. Pin it explicitly.
TORCH_THREADS = max(1, int(os.getenv("TORCH_THREADS", "2")))
torch.set_num_threads(TORCH_THREADS)

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

# v8 is not a new set of weights. It is v6 and v7b run together, each head's raw
# score mapped through that model's empirical CDF, the two averaged, passed
# through an isotonic fit and forced monotone across heads. The two models
# trained on disjoint data and fail on different images, which is why averaging
# them beats either alone: on the held-out half of APTOS (1,831 images, used for
# nothing else) referable-DR AUC is 0.940 against v6's 0.916 and v7b's 0.899,
# +0.024 [+0.017, +0.031] and +0.041 [+0.032, +0.050], both p<0.001 paired.
#
# The calibration is the point. v7b alone ranks well but its referable positives
# sit at median 0.018, so it needs a 2.1e-4 threshold; dropped into this module
# under V6_THRESH it would have reported 9.2% of referable cases as grade >= 2
# and never returned grade 4 at all - silently, with no error. Calibrated, the
# operating points are 0.731 and 0.187 and expected calibration error on the
# referable head is 0.021 against v6's 0.184.
# The numeric ICDR grade is NOT reported. v8's grade is right 2.6% of the time
# outside grades 0 and 2: on 1,831 held-out images it recovered 2 of 189 grade-1
# cases, 8 of 98 grade-3 and 1 of 142 grade-4, reporting true proliferative
# disease as "Moderate" in 109 of 142. The heads cannot rank severity either -
# P(a grade-3 image scores above a grade-2 one) is 0.474, a coin flip - so no
# threshold, cut-point or probability-banding scheme recovers it. All of that was
# measured; five separate attempts are recorded in HANDOVER.md.
#
# The cause is supervision, not architecture: severity is trained on Paraguay
# alone, 1,437 images containing five grade-1 examples. Until graded data exists
# the honest output is the referral decision, which is sound - AUC 0.940, 97.9%
# sensitivity. Set GRADE_REPORTING=1 to surface the grade anyway; it is off
# because a "Moderate" label on a proliferative eye understates urgency in the
# one direction that harms a patient.
GRADE_REPORTING = str(os.getenv("GRADE_REPORTING", "0")).strip() == "1"
V7B_WEIGHTS = os.path.join(MODEL_DIR, "alati_dr_v7b.pth")
V8_CALIBRATION = os.path.join(MODEL_DIR, "v8_calibration.json")
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

# Attribution belongs to the model that is loaded, not to the module. Returning
# the CC BY credits unconditionally meant /model_info credited Paraguay and
# RFMiD 2.0 while serving v1, which was trained on neither - a false provenance
# claim on a public endpoint, and in the one direction that matters here.
# A licence is asserted only where one is established at source.
SOURCE_ATTRIBUTION = {
    "paraguay": "Paraguay fundus dataset (Benitez et al.), CC BY 4.0, doi:10.5281/zenodo.4891308",
    "rfmid2": "RFMiD 2.0 (Panchal, Naik, Kokare, Pachade et al.), CC BY 4.0, doi:10.5281/zenodo.7505822",
    "rfmid": "RFMiD 1.0 (Pachade, Porwal, Kokare et al.), doi:10.3390/data6020014",
}


def _attribution_for(sources) -> list:
    """Credits for the sources the loaded model was actually trained on."""
    return [SOURCE_ATTRIBUTION[k] for k in sorted(sources or []) if k in SOURCE_ATTRIBUTION]


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


def _load_ordinal(path):
    if not os.path.exists(path):
        raise RuntimeError(f"weights not found for {MODEL_VERSION}: {path}")
    ck = torch.load(path, map_location=DEVICE)
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
    sha = _sha256_bytes(open(path, "rb").read())
    return (m, tf, ck.get("size", 768), sha, os.path.getsize(path),
            list(ck.get("sources", [])))


V6_MODEL = V6_TRANSFORM = None
V6_SIZE = V6_SHA = V6_BYTES = None
V6_SOURCES = []
V8_MODELS = {}
V8_TRANSFORMS = {}
V8_CAL = None
V8_THRESH = []
# Whether the severity heads were actually trained. A model trained only on
# binary-labelled data has P(grade>=2..4) masked out of its loss for every
# positive, so those heads never see a positive example and emit noise - on
# APTOS that produced an inverted AUC of 0.086. Grades are reported only when a
# graded source was in the training mix.
SEVERITY_TRAINED = False
GRADED_SOURCES = {"paraguay"}
if MODEL_VERSION == "v6":
    V6_MODEL, V6_TRANSFORM, V6_SIZE, V6_SHA, V6_BYTES, V6_SOURCES = _load_ordinal(V6_WEIGHTS)
    SEVERITY_TRAINED = bool(GRADED_SOURCES & set(V6_SOURCES))
elif MODEL_VERSION == "v8":
    with open(V8_CALIBRATION, "r", encoding="utf-8") as fh:
        V8_CAL = json.load(fh)
    for _key, _path in (("v6", V6_WEIGHTS), ("v7b", V7B_WEIGHTS)):
        _m, _tf, _size, _sha, _bytes, _src = _load_ordinal(_path)
        # THE GUARD. A calibration curve belongs to the exact weights it was fit
        # against; pair it with different ones and every threshold below is
        # meaningless while the service keeps answering confidently. This is the
        # failure that a v7b drop-in would have caused, so it refuses to start
        # rather than serve. Same reasoning as SEVERITY_TRAINED, one level up.
        _want = V8_CAL["weights"][_key]["sha256"]
        if _sha != _want:
            raise RuntimeError(
                f"v8 calibration/weights mismatch for {_key}: {os.path.basename(_path)} "
                f"is sha256 {_sha[:16]}, calibration was fit against {_want[:16]}. "
                f"Refusing to start - thresholds from this artifact do not describe "
                f"these weights."
            )
        V8_MODELS[_key], V8_TRANSFORMS[_key] = _m, _tf
        if _key == "v6":
            V6_SIZE, V6_SHA, V6_BYTES, V6_SOURCES = _size, _sha, _bytes, _src
        else:
            V6_SOURCES = sorted(set(V6_SOURCES) | set(_src))
    V8_THRESH = [float(V8_CAL["thresholds"][str(h)]) for h in range(4)]
    SEVERITY_TRAINED = bool(GRADED_SOURCES & set(V6_SOURCES))
    if os.getenv("V6_THRESH"):
        # V6_THRESH is v6's operating point and nothing else's. Honouring it here
        # is how the 9.2% failure would have happened.
        raise RuntimeError("V6_THRESH is set but MODEL_VERSION=v8; v8 takes its "
                           "thresholds from v8_calibration.json. Unset V6_THRESH.")


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

    # The severity heads are only trustworthy on a model whose training data
    # carried real grades. Where they were masked out (a binary-labelled source),
    # SEVERITY_TRAINED is false and the grade is withheld rather than shown -
    # an untrained ordinal head still emits a number, and that number is noise.
    if is_dr and SEVERITY_TRAINED and grade >= 1:
        display = f"Diabetic Retinopathy — {GRADE_LABEL[grade]} (grade {grade})"
    else:
        display = translate_code(code)

    return {
        "phase": ACTIVE_PHASE,
        "phase_name": PHASE_NAME,
        "model_version": MODEL_VERSION,
        "top_code": code,
        "top_prob": p_any,
        "top3": [("D", p_any), ("N", 1.0 - p_any)],
        "final_code": code,
        "final_reason": f"{MODEL_VERSION} ordinal: P(any DR)={p_any:.3f} vs threshold {V6_THRESH:.3f}",
        "translated": translate_code(code),
        "display": display,
        "probs": {"N": 1.0 - p_any, "D": p_any},
        "confidence": p_any if is_dr else 1.0 - p_any,
        "enhanced": False,
        # new here - v1 has no notion of severity at all
        "grade": grade if SEVERITY_TRAINED else None,
        "grade_label": GRADE_LABEL.get(grade, "Unknown") if SEVERITY_TRAINED else None,
        "severity_available": SEVERITY_TRAINED,
        "p_any_dr": p_any,
        "p_referable": p_ref,
        "ordinal": [float(v) for v in p],
    }


def _cdf(cal, x):
    """Empirical CDF lookup. cal is (breakpoints, values), both ascending."""
    bp, val = cal
    return float(np.interp(x, bp, val, left=0.0, right=1.0))


def _pav(cal, x):
    """Isotonic (pool-adjacent-violators) lookup: piecewise constant, ascending."""
    bp, val = cal
    i = int(np.searchsorted(np.asarray(bp), x, side="left"))
    return float(val[min(i, len(val) - 1)])


def _v8_scores(img: Image.Image) -> List[float]:
    """Calibrated, monotone P(grade >= 1..4) from the v6+v7b ensemble."""
    per_model = []
    for key in ("v6", "v7b"):
        x = V8_TRANSFORMS[key](img).unsqueeze(0)
        batch = torch.cat([x, torch.flip(x, dims=[3])], dim=0) if MIRROR_TTA else x
        with torch.no_grad():
            raw = torch.sigmoid(V8_MODELS[key](batch)).mean(dim=0).tolist()
        per_model.append([_cdf(V8_CAL["cdf"][key][h], raw[h]) for h in range(4)])
    ens = [(per_model[0][h] + per_model[1][h]) / 2.0 for h in range(4)]
    cal = [_pav(V8_CAL["isotonic"][str(h)], ens[h]) for h in range(4)]
    out, cur = [], 1.0
    for v in cal:                      # P(>=1) >= P(>=2) >= P(>=3) >= P(>=4)
        cur = min(cur, v)
        out.append(cur)
    return out


def _v8_predict(image_bytes: bytes) -> dict:
    """Same return shape as _v6_predict, so no caller changes."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = _square_pad(_crop_to_disc(img))
    p = _v8_scores(img)
    p_any, p_ref = float(p[0]), float(p[1])

    referred = p_ref >= V8_THRESH[1]
    # A referral that came back "no DR" would be incoherent, so the referral head
    # also forces the binary call. This is the deployed rule and it is the rule
    # that was measured - the held-out figures below describe this exact
    # disjunction, not the head-0 threshold on its own.
    is_dr = (p_any >= V8_THRESH[0]) or referred

    # Expected grade: round the sum of the calibrated head probabilities. Chosen
    # over per-head thresholding, which cost 24 points of within-one agreement by
    # picking aggressive cut-offs on the rare high grades.
    grade = int(min(4, max(0, round(sum(p)))))
    if referred:
        grade = max(grade, 2)

    code = "D" if is_dr else "N"
    # Referral, not severity. "not referable" still means disease is present and
    # should be monitored - it is not a clean result.
    if not is_dr:
        display = translate_code(code)
    elif referred:
        display = "Diabetic retinopathy — referable"
    else:
        display = "Diabetic retinopathy — not referable"
    if is_dr and GRADE_REPORTING and SEVERITY_TRAINED:
        display += f" (grade {grade})"

    return {
        "phase": ACTIVE_PHASE,
        "phase_name": PHASE_NAME,
        "model_version": "v8",
        "top_code": code,
        "top_prob": p_any,
        "top3": [("D", p_any), ("N", 1.0 - p_any)],
        "final_code": code,
        "final_reason": (f"v8 ensemble: P(any DR)={p_any:.3f} vs {V8_THRESH[0]:.3f}, "
                         f"P(referable)={p_ref:.3f} vs {V8_THRESH[1]:.3f}"),
        "translated": translate_code(code),
        "display": display,
        "probs": {"N": 1.0 - p_any, "D": p_any},
        "confidence": p_any if is_dr else 1.0 - p_any,
        "enhanced": False,
        "grade": grade if (GRADE_REPORTING and SEVERITY_TRAINED) else None,
        "grade_label": (GRADE_LABEL.get(grade, "Unknown")
                        if (GRADE_REPORTING and SEVERITY_TRAINED) else None),
        "severity_available": GRADE_REPORTING and SEVERITY_TRAINED,
        "grade_withheld_reason": (None if GRADE_REPORTING else
                                  "ICDR grade is not reported: the severity heads cannot "
                                  "rank grades above 2 (P(3>2)=0.47). Use the referral "
                                  "decision, AUC 0.940."),
        "referable": referred,
        "p_any_dr": p_any,
        "p_referable": p_ref,
        "referred": referred,
        "ordinal": [float(v) for v in p],
    }


def model_info() -> dict:
    """
    What is actually running. Unauthenticated on purpose: after the 4 September
    deploy failure there was no way to tell from outside whether the service was
    serving v6 or had fallen back to v1, and MODEL_VERSION is an env var that can
    change without a commit. No patient data, no secrets - just which weights are
    loaded and at what thresholds.
    """
    info = {
        "model_version": MODEL_VERSION,
        "build_marker": BUILD_MARKER,
        # What the service actually reports, not what the weights could support.
        # SEVERITY_TRAINED says a graded source was in the mix; it does not say
        # the grade is fit to publish, and with GRADE_REPORTING off it is not.
        "severity_available": GRADE_REPORTING and SEVERITY_TRAINED,
        "severity_trained": SEVERITY_TRAINED,
        "trained_on": V6_SOURCES,
        "data_attribution": _attribution_for(V6_SOURCES),
        "mirror_tta": MIRROR_TTA,
    }
    if MODEL_VERSION == "v8":
        info.update({
            "ensemble": V8_CAL.get("models"),
            "weights_sha": {k: v["sha256"][:16] for k, v in V8_CAL["weights"].items()},
            "thresholds": {"any_dr": V8_THRESH[0], "referable": V8_THRESH[1]},
            "grade_rule": V8_CAL.get("grade_rule"),
            "grade_reported": GRADE_REPORTING,
            "calibrated_on": V8_CAL.get("calibrated_on"),
            "measured": V8_CAL.get("measured"),
        })
    elif MODEL_VERSION == "v6":
        info.update({"weights_sha": V6_SHA[:16] if V6_SHA else None,
                     "thresholds": {"any_dr": V6_THRESH}})
    else:
        info.update({"weights_sha": WEIGHTS_SHA[:16] if WEIGHTS_SHA else None,
                     "active_variant": ACTIVE_VARIANT})
    return info


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
    if MODEL_VERSION == "v8":
        return _v8_predict(image_bytes)
    if MODEL_VERSION == "v6":
        # These were trained on cropped, padded, ImageNet-normalized images. The v1
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
    if MODEL_VERSION == "v8":
        return {"build_marker": BUILD_MARKER, "load_mode": "strict",
                "active_variant": "v6+v7b calibrated ensemble", **model_info(), **raw}
    if MODEL_VERSION == "v6":
        return {
            "build_marker": BUILD_MARKER,
            "model_version": MODEL_VERSION,
            "active_variant": "resnet50-ordinal-768",
            "load_mode": "strict",
            "weights_sha": V6_SHA,
            "weights_size": V6_BYTES,
            "input_size": V6_SIZE,
            "threshold": V6_THRESH,
            "trained_on": V6_SOURCES,
            "severity_available": SEVERITY_TRAINED,
            "data_attribution": _attribution_for(V6_SOURCES),
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