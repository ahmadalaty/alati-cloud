"""
Image-quality gate: is this a gradable colour fundus photograph at all?

Without it, every input gets a diagnosis. On 10 Sep 2026 both v8 and v10 read an
all-black image as "Normal" / "Other abnormality" and a frame of random noise as
"Diabetic retinopathy - referable". A screening tool must be able to say "I
cannot read this photo" - a confident answer on an unreadable image is the
failure a patient never sees.

Deliberately simple and inspectable: a handful of image statistics, each with a
threshold, each with a plain-language reason. No learned component, because the
only data it could learn "bad" from would be whatever negatives happened to be
on disk. Thresholds were set on one pool of images and checked on another
(D:\\alati-train\\gate_eval.py); the false-reject rate on real photographs is
the number that matters, and it is published in /model_info.

Only PIL and numpy - the same code runs in the service and in the evaluation.
"""
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

# Thresholds: D:\alati-train\gate_select2.py ("rule 2"). Each bound lies a margin
# beyond the most extreme of 9,824 real photographs in the tuning pool, so the
# gate refuses only what no camera produced there; borderline images go to the
# model. The first rule (0.1th/99.9th percentiles) wrongly refused 4.7% of one
# hospital's photos and was replaced.
MIN_SIDE = 256            # px, shorter side of the upload
MIN_RETINA_FRAC = 0.195   # share of the (cropped, squared) frame that is non-black
MIN_LUM = 15.3            # mean luminance inside the retina, 0-255
MAX_LUM = 218.1
GRAY_ABS_RED_BLUE = 0.01  # |R - B| / luminance below this = grayscale: OCT, masks, scans, noise
MIN_SHARP = 0.79          # Laplacian std, green channel, 512px frame, well inside the rim
MAX_SHARP = 21.25         # above this the "detail" is noise, not anatomy


def _crop_to_disc(img: Image.Image) -> Image.Image:
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
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (0, 0, 0))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def _erode(mask: np.ndarray, k: int) -> np.ndarray:
    """Binary erosion with a k x k square, edges replicated.

    Pixel-identical to PIL's ImageFilter.MinFilter(k) on these masks (0 of
    14 x 512 x 512 pixels differ, fundus and full-frame noise alike), and ~100x
    faster: MinFilter(41) alone was ~0.25-0.8 s of every scan."""
    r = k // 2
    p = np.pad(mask.astype(np.int32), r, mode="edge")
    ii = np.zeros((p.shape[0] + 1, p.shape[1] + 1), np.int64)
    ii[1:, 1:] = p.cumsum(0).cumsum(1)
    box = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return box == k * k


def quality_features(img: Image.Image) -> Dict[str, float]:
    img = img.convert("RGB")
    w, h = img.size
    sq_full = _square_pad(_crop_to_disc(img))
    sq = sq_full.resize((256, 256), Image.BILINEAR)
    a = np.asarray(sq, dtype=np.float32)
    mask = a.max(axis=2) > 18
    frac = float(mask.mean())
    if mask.sum() < 50:
        return {"min_side": float(min(w, h)), "retina_frac": frac, "lum": 0.0,
                "red_blue": 0.0, "sharp": 0.0}
    r, g, b = a[..., 0][mask], a[..., 1][mask], a[..., 2][mask]
    lum = float((0.299 * r + 0.587 * g + 0.114 * b).mean())
    red_blue = float((r.mean() - b.mean()) / (lum + 1.0))
    # Sharpness: Laplacian of the green channel at 512px, measured only well inside
    # the retina. At 256px with a thin margin the resampled rim leaked in as fake
    # "detail" and flat fields read as textured. This version catches flat and
    # detail-free frames; it does NOT catch moderate blur - at thresholds that
    # spare real photos, 0 of 100 test images blurred at radius 12 were refused.
    big = sq_full.resize((512, 512), Image.BILINEAR)
    g512 = np.asarray(big, dtype=np.float32)
    m512 = g512.max(axis=2) > 18
    lap = np.asarray(Image.fromarray(g512[..., 1].astype(np.uint8)).filter(
        ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 128)), dtype=np.float32) - 128.0
    inner = _erode(m512, 41)
    sharp = float(lap[inner].std()) if inner.sum() > 200 else 0.0
    return {"min_side": float(min(w, h)), "retina_frac": frac, "lum": lum,
            "red_blue": red_blue, "sharp": sharp}


def check_quality(img: Image.Image) -> Tuple[bool, Optional[str], Dict[str, float]]:
    """(ok, reason-if-not, features). Reasons are written for the person holding the camera."""
    f = quality_features(img)
    if f["min_side"] < MIN_SIDE:
        return False, "The image is too small to grade. Upload the original photo.", f
    if f["retina_frac"] < MIN_RETINA_FRAC or f["lum"] < MIN_LUM:
        return False, "The image is too dark to see the retina. Retake the photo.", f
    if f["lum"] > MAX_LUM:
        return False, "The image is overexposed. Retake the photo with less flash.", f
    if abs(f["red_blue"]) < GRAY_ABS_RED_BLUE:
        # grayscale scans (OCT, masks) and random noise both land here
        return False, "This does not look like a colour fundus photograph.", f
    if f["sharp"] < MIN_SHARP:
        return False, "The image is too blurred to grade. Refocus and retake the photo.", f
    if f["sharp"] > MAX_SHARP:
        return False, "This does not look like a fundus photograph (image is mostly noise).", f
    return True, None, f
