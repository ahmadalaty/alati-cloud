import os
import json
from collections import OrderedDict
from typing import Dict, Any, Tuple, List

import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image


BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model_files")

MODEL_18_PATH = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet18.pth")
MODEL_50_PATH = os.path.join(MODEL_DIR, "alati_dualeye_model_resnet50.pth")
LABELS_PATH   = os.path.join(MODEL_DIR, "labels.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _label_to_str(x) -> str:
    """
    Accept labels.json formats:
      1) ["normal","dr",...]
      2) [{"code":"N","name":"normal"}, ...]
      3) [{"label":"normal"}, ...]
    Returns a safe string label.
    """
    if isinstance(x, str):
        return x

    if isinstance(x, dict):
        # Prefer most informative keys if present
        for k in ("name", "label", "diagnosis", "class", "title"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # If only "code" exists, use it
        v = x.get("code")
        if isinstance(v, str) and v.strip():
            return v.strip()

        # fallback dict -> string
        return json.dumps(x, ensure_ascii=False)

    # fallback any type -> string
    return str(x)


def load_labels() -> List[str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_raw = json.load(f)

    if not isinstance(labels_raw, list) or not labels_raw:
        raise RuntimeError("labels.json must be a non-empty JSON list.")

    labels = [_label_to_str(x) for x in labels_raw]

    # ensure unique + non-empty
    labels = [x if x else f"class_{i}" for i, x in enumerate(labels)]
    return labels


LABELS = load_labels()
NUM_CLASSES = len(LABELS)


TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def _extract_state_dict(loaded_obj) -> OrderedDict:
    if isinstance(loaded_obj, OrderedDict):
        return loaded_obj

    if isinstance(loaded_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in loaded_obj and isinstance(loaded_obj[key], (dict, OrderedDict)):
                sd = loaded_obj[key]
                return OrderedDict(sd) if not isinstance(sd, OrderedDict) else sd

        if any(isinstance(v, torch.Tensor) for v in loaded_obj.values()):
            return OrderedDict(loaded_obj)

    raise RuntimeError(
        "Unsupported .pth format. Expected state_dict (OrderedDict/dict) or checkpoint containing state_dict."
    )


def _strip_module_prefix(state_dict: OrderedDict) -> OrderedDict:
    if not state_dict:
        return state_dict

    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k.replace("module.", "", 1)] = v
        return new_sd
    return state_dict


def _build_model(variant: str):
    v = (variant or "resnet18").strip().lower()

    if v == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
        return model, "resnet50"

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, NUM_CLASSES)
    return model, "resnet18"


def _load_model_weights(model: torch.nn.Module, weights_path: str):
    loaded = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(loaded)
    state_dict = _strip_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()

    if missing:
        print(f"[inference] WARNING missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[inference] WARNING unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")

    return model


_CACHED_MODEL = None
_CACHED_VARIANT = None


def get_model(variant: str = "resnet18") -> Tuple[torch.nn.Module, str]:
    global _CACHED_MODEL, _CACHED_VARIANT

    v = (variant or "resnet18").strip().lower()
    if v not in ("resnet18", "resnet50"):
        v = "resnet18"

    if _CACHED_MODEL is not None and _CACHED_VARIANT == v:
        return _CACHED_MODEL, _CACHED_VARIANT

    model, chosen = _build_model(v)
    weights_path = MODEL_50_PATH if chosen == "resnet50" else MODEL_18_PATH

    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}")

    _CACHED_MODEL = _load_model_weights(model, weights_path)
    _CACHED_VARIANT = chosen
    return _CACHED_MODEL, _CACHED_VARIANT


def run_inference(image_path: str, eye: str = "left", model_variant: str = "resnet18") -> Dict[str, Any]:
    model, chosen_variant = get_model(model_variant)

    img = Image.open(image_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu()

    probs_dict = {str(LABELS[i]): float(probs[i].item()) for i in range(NUM_CLASSES)}
    top_label = max(probs_dict, key=probs_dict.get)
    top_prob = probs_dict[top_label]

    triage = "Refer" if top_prob >= 0.6 else "Observe"

    return {
        "model_variant": chosen_variant,
        "eye": (eye or "").lower(),
        "probs": probs_dict,
        "top_label": top_label,
        "top_prob": float(top_prob),
        "triage": triage,
    }
