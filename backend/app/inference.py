import json
import os
import torch
import torchvision.transforms as T
from PIL import Image

# -------------------------
# Paths / labels
# -------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

with open(os.path.join(MODEL_DIR, "labels.json"), "r", encoding="utf-8") as f:
    LABELS = json.load(f)

NUM_CLASSES = len(LABELS)

# Default model variant (you can switch later via env)
DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower()


# -------------------------
# Helpers
# -------------------------
def _clean_state_dict(state: dict) -> dict:
    """
    Your checkpoint uses keys like 'backbone.conv1.weight' (and sometimes 'module.backbone...')
    We convert them into torchvision ResNet keys ('conv1.weight').
    """
    cleaned = {}
    for k, v in state.items():
        nk = k

        # common wrappers
        if nk.startswith("module."):
            nk = nk[len("module."):]

        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]

        # if you used 'model.' wrapper sometimes
        if nk.startswith("model."):
            nk = nk[len("model."):]

        cleaned[nk] = v
    return cleaned


def _load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")

    # if saved as {"state_dict": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    return ckpt


# -------------------------
# Model building/loading
# -------------------------
def load_model(model_variant: str):
    model_variant = (model_variant or "resnet18").strip().lower()

    if model_variant == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        weights_path = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
        model_variant = "resnet18"

    # force FC to match our labels count
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    ckpt = _load_checkpoint(weights_path)

    # Case A: checkpoint is a full model object
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, model_variant

    # Case B: checkpoint is a state_dict
    state = _clean_state_dict(ckpt)

    # load strictly (should match after cleaning). If you changed heads, we allow non-strict for fc.
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # allow head mismatch only (fc.*)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, model_variant


MODEL, ACTIVE_VARIANT = load_model(DEFAULT_VARIANT)


# -------------------------
# Preprocessing (standard ImageNet)
# -------------------------
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# -------------------------
# Inference
# -------------------------
def run_inference(image_path: str, eye: str):
    """
    image_path: image file path
    eye: 'left' or 'right' (kept for future eye-specific pipelines)
    """
    eye = (eye or "left").strip().lower()
    if eye not in ("left", "right"):
        eye = "left"

    img = Image.open(image_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.sigmoid(logits)[0]

    probs_dict = {LABELS[i]: float(probs[i].item()) for i in range(NUM_CLASSES)}
    top_label = max(probs_dict, key=probs_dict.get)
    top_prob = probs_dict[top_label]

    triage = "Refer" if top_prob >= 0.5 else "Routine"

    return {
        "model_variant": ACTIVE_VARIANT,
        "eye": eye,
        "probs": probs_dict,
        "top_label": top_label,
        "top_prob": top_prob,
        "triage": triage,
    }
