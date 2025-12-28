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

LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

NUM_CLASSES = len(LABELS)

DEFAULT_VARIANT = os.getenv("MODEL_VARIANT", "resnet18").strip().lower()


# -------------------------
# Helpers
# -------------------------
def _clean_state_dict(state: dict) -> dict:
    """
    Converts keys like:
      'module.backbone.conv1.weight' -> 'conv1.weight'
      'backbone.layer1.0...' -> 'layer1.0...'
    """
    cleaned = {}
    for k, v in state.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module.") :]

        if nk.startswith("backbone."):
            nk = nk[len("backbone.") :]

        if nk.startswith("model."):
            nk = nk[len("model.") :]

        cleaned[nk] = v
    return cleaned


def _extract_state_dict(ckpt):
    """
    Handles:
    - state_dict directly (dict of tensors)
    - dict containing "state_dict"
    - other common wrappers
    """
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        # sometimes it's already a state dict
        return ckpt
    return None


# -------------------------
# Model loading
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

    # set classifier head
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

    ckpt = torch.load(weights_path, map_location="cpu")

    # if someone saved full model object
    if not isinstance(ckpt, dict):
        ckpt.eval()
        return ckpt, model_variant

    state = _extract_state_dict(ckpt)
    if state is None:
        raise RuntimeError("Checkpoint format not understood (no state_dict found)")

    state = _clean_state_dict(state)

    # first try strict
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # allow only head mismatches
        model.load_state_dict(state, strict=False)

    model.eval()
    return model, model_variant


MODEL, ACTIVE_VARIANT = load_model(DEFAULT_VARIANT)


# -------------------------
# Preprocessing
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
