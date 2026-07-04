# style_features.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray

try:
    from sklearn.metrics import silhouette_score  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


@dataclass
class StyleAnalysis:
    edge_density: float
    edge_coherence: float
    edge_thickness_score: float
    saturation_mean: float
    saturation_std: float
    color_clusters: int
    color_silhouette: float
    contrast: float
    grayscale_ratio: float
    grain_score: float
    hf_ratio: float
    dot_pattern_score: float
    straight_line_score: float
    brush_texture_score: float
    bokeh_score: float
    class_scores: Dict[str, float]
    primary_class: str


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img

def _downscale(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    ms = max(h, w)
    if ms <= max_side:
        return img
    s = max_side / ms
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

def _edge_metrics(bgr: np.ndarray) -> Tuple[float, float, float]:
    # Kanten-Dichte, Orientierungs-Kohärenz (Proxy), und Kanten-"Dicke"
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _downscale(gray, 640)
    edges = canny(gray.astype(np.float32) / 255.0, sigma=1.2)
    edge_density = float(edges.mean())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) + 1e-6
    ori = np.arctan2(gy, gx)

    mask = (mag > (mag.mean() + mag.std()))
    if int(mask.sum()) > 100:
        ori_edges = ori[mask]
        var_ori = float(np.var(np.sin(ori_edges)) + np.var(np.cos(ori_edges)))
        edge_coherence = max(0.0, 1.0 - min(1.0, var_ori / 1.0))
    else:
        edge_coherence = 0.0

    edges_u = (edges.astype(np.uint8) * 255)
    dil = cv2.dilate(edges_u, np.ones((3, 3), np.uint8), 1)
    thickness_score = float((dil > 0).mean() - edge_density)
    return edge_density, edge_coherence, thickness_score

def _estimate_silhouette_fallback(X: np.ndarray, labels: np.ndarray) -> float:
    # Lightweight Silhouette-Approximation ohne sklearn
    try:
        labs = labels.reshape(-1)
        ks = np.unique(labs)
        if ks.size < 2:
            return -0.05
        cents = []
        intras = []
        for k in ks:
            pts = X[labs == k]
            if pts.size == 0:
                continue
            c = pts.mean(axis=0)
            cents.append(c)
            d = np.linalg.norm(pts - c, axis=1).mean()
            intras.append(d if np.isfinite(d) else 0.0)
        if len(cents) < 2:
            return -0.03
        cents = np.vstack(cents)
        dsum = 0.0
        cnt = 0
        for i in range(len(cents)):
            for j in range(i + 1, len(cents)):
                dsum += float(np.linalg.norm(cents[i] - cents[j]))
                cnt += 1
        inter = (dsum / cnt) if cnt else 0.0
        intra = float(np.mean(intras)) if intras else 0.0
        if intra <= 1e-6:
            return 0.2
        ratio = inter / intra
        val = (ratio - 1.0) * 0.15
        return float(np.clip(val, -0.1, 0.5))
    except Exception:
        return -0.05

def _color_metrics(bgr: np.ndarray) -> Tuple[float, float, int, float]:
    # Sättigungs-Statistik + einfache KMeans-Paletten-Clustering-Qualität
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    s_mean = float(s.mean())
    s_std = float(s.std())

    sample = _downscale(bgr, 480)
    flat = sample.reshape(-1, 3).astype(np.float32)
    if flat.shape[0] > 40000:
        idx = np.random.choice(flat.shape[0], 40000, replace=False)
        flat = flat[idx]

    best_k = 3
    best_sil = -1.0
    prev_compact = None
    for k in range(3, 9):
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 1.0)
        compactness, labels, _ = cv2.kmeans(flat, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
        if _HAVE_SKLEARN:
            try:
                if flat.shape[0] > 2000:
                    idx = np.random.choice(flat.shape[0], 2000, replace=False)
                    sil = silhouette_score(flat[idx], labels[idx].ravel(), metric='euclidean')  # type: ignore
                else:
                    sil = silhouette_score(flat, labels.ravel(), metric='euclidean')  # type: ignore
            except Exception:
                sil = -1.0
        else:
            sil = _estimate_silhouette_fallback(flat, labels)

        if sil > best_sil:
            best_sil = sil
            best_k = k

        if prev_compact is None:
            prev_compact = compactness
        else:
            if compactness > prev_compact * 0.98:
                break
            prev_compact = compactness

    return s_mean, s_std, int(best_k), float(best_sil)

def _contrast_metric(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(gray.std())

def _grayscale_ratio(bgr: np.ndarray) -> float:
    b, g, r = cv2.split(bgr.astype(np.float32))
    diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / (3.0 * 255.0)
    return float((diff < 0.03).mean())

def _grain_and_hf(bgr: np.ndarray) -> Tuple[float, float]:
    # Korn/Noise via Laplacian-Varianz; HF/LF Spektrum via FFT-Ringe
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    g_small = _downscale((gray * 255).astype(np.uint8), 512).astype(np.float32) / 255.0
    lap = cv2.Laplacian(g_small, cv2.CV_32F)
    grain = float(lap.var())

    G = _downscale(gray, 512)
    F = np.fft.fftshift(np.fft.fft2(G))
    mag = np.log1p(np.abs(F))
    H, W = mag.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    r = np.hypot(yy - cy, xx - cx)
    hf_mask = (r > min(H, W) * 0.22) & (r < min(H, W) * 0.48)
    lf_mask = (r < min(H, W) * 0.12)
    hf_ratio = float(mag[hf_mask].mean() / (mag[lf_mask].mean() + 1e-6))
    return grain, hf_ratio

def _dot_pattern_score(bgr: np.ndarray) -> float:
    # Halftone-Ring-Detektor (Frequenzbereich)
    gray = rgb2gray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    gray = _downscale((gray * 255).astype(np.uint8), 512).astype(np.float32) / 255.0
    F = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(F))
    H, W = mag.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    r = np.hypot(yy - cy, xx - cx)
    ring = (r > 26) & (r < 46)
    neigh = ((r > 18) & (r < 24)) | ((r > 48) & (r < 56))
    ring_mean = float(mag[ring].mean())
    neigh_mean = float(mag[neigh].mean() + 1e-6)
    z = (ring_mean - neigh_mean) / (neigh_mean + 1e-6)
    return float(max(0.0, z * 3.0))

def _straight_line_score(bgr: np.ndarray) -> float:
    # Hough-Linienlänge relativ zum Umfang
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _downscale(gray, 800)
    edges = cv2.Canny(gray, 80, 160, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=3)
    if lines is None or len(lines) == 0:
        return 0.0
    h, w = gray.shape
    per = float(h + w)
    total_len = 0.0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        total_len += float(np.hypot(x2 - x1, y2 - y1))
    return float(min(1.0, total_len / (per * 15.0)))

def _brush_texture_score(bgr: np.ndarray) -> float:
    # DoG-Varianz als grober Pinseltextur-Hinweis
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    Ls = _downscale(L, 512)
    g1 = cv2.GaussianBlur(Ls, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(Ls, (0, 0), 3.0)
    dog = cv2.absdiff(g1, g2)
    mean = cv2.blur(dog, (9, 9))
    sq = cv2.blur(dog * dog, (9, 9))
    var = sq - mean * mean
    return float(np.clip(var.mean() * 50.0, 0.0, 1.0))

def _bokeh_score(bgr: np.ndarray) -> float:
    # Varianz lokaler Schärfe → Hinweis auf DoF/Bokeh
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gs = _downscale(gray, 512)
    lap = cv2.Laplacian(gs, cv2.CV_32F)
    sharp = cv2.GaussianBlur(lap * lap, (0, 0), 1.0)
    h, w = sharp.shape
    tiles = 6
    th, tw = h // tiles, w // tiles
    vals = []
    for i in range(tiles):
        for j in range(tiles):
            patch = sharp[i*th:(i+1)*th, j*tw:(j+1)*tw]
            if patch.size > 0:
                vals.append(float(patch.mean()))
    vals = np.array(vals, dtype=np.float32)
    if vals.size < 4:
        return 0.0
    v = float(np.std(vals))
    return float(np.clip(v * 10.0, 0.0, 1.5))


def analyze_style(path: Path) -> StyleAnalysis:
    bgr = _load_bgr(path)

    ed, ecoh, th = _edge_metrics(bgr)
    s_mean, s_std, k, sil = _color_metrics(bgr)
    contr = _contrast_metric(bgr)
    gray_ratio = _grayscale_ratio(bgr)
    grain, hf_ratio = _grain_and_hf(bgr)
    dots = _dot_pattern_score(bgr)
    straight_score = _straight_line_score(bgr)
    brush_score = _brush_texture_score(bgr)
    bokeh = _bokeh_score(bgr)

    # Photo
    photo = 0.0
    if hf_ratio > 1.15:
        photo += 0.55
    if bokeh > 0.26:
        photo += 0.30
    if s_mean > 0.20 and s_std > 0.09:
        photo += 0.18
    if k >= 7:
        photo += 0.12
    if sil < 0.05:
        photo += 0.08
    if (hf_ratio > 1.15) and (bokeh > 0.26):
        photo += 0.12
    if contr > 0.17:
        photo += 0.07
    if grain > 80.0:
        photo += 0.05

    # Photo-like gate (moderate HF, natural palette, low line structure)
    photo_like_gate = (
        (1.07 <= hf_ratio <= 1.15) and
        (bokeh >= 0.18 or contr >= 0.15) and
        ((k >= 6) or (sil <= 0.08)) and
        (straight_score < 0.18) and
        not (ecoh > 0.28 and 0.012 < th < 0.060) and
        (dots < 0.8)
    )
    if photo_like_gate:
        photo += 0.20

    # Relaxed photo-like gate (Policy): bestimmte posterartige Fälle als "photo"
    photo_like_gate_relaxed = (
        (hf_ratio < 1.00) and
        (bokeh < 0.12) and
        (k <= 4) and (sil >= 0.45) and
        (s_mean >= 0.32) and (s_std >= 0.18) and
        (straight_score < 0.12) and
        (ecoh <= 0.06) and
        (th >= 0.10) and
        (dots < 0.6)
    )
    if photo_like_gate_relaxed:
        photo += 0.32  # starker Boost, damit "photo" gewinnt

    # Anti-photo Penalties
    if ecoh > 0.32 and 0.012 < th < 0.050 and bokeh < 0.22 and hf_ratio < 1.14:
        photo -= 0.35
    if (ecoh <= 0.06 and th >= 0.10 and hf_ratio < 0.95 and k <= 4 and sil >= 0.45):
        photo -= 0.05
    if dots > 1.1 and gray_ratio > 0.45:
        photo -= 0.20

    photo = max(0.0, photo)

    # Comic (UNVERÄNDERT LASSEN)
    comic = 0.0
    if ecoh >= 0.20 and 0.012 < th < 0.060 and hf_ratio < 1.12:
        comic += 0.50
    if bokeh < 0.22:
        comic += 0.12
    if k <= 7 and sil > 0.06:
        comic += 0.10
    if ed > 0.040:
        comic += 0.06
    if dots > 0.9:
        comic += 0.08
    if (ecoh <= 0.05 and k <= 4 and sil >= 0.45 and th >= 0.10 and hf_ratio < 0.95):
        comic -= 0.16
    if (hf_ratio > 1.15 and bokeh > 0.26) or (s_mean > 0.22 and s_std > 0.10 and k >= 7):
        comic -= 0.08
    if s_mean > 0.28 and s_std > 0.12 and k >= 7:
        comic -= 0.10
    if (0.16 <= s_mean <= 0.42) and (0.12 <= s_std <= 0.35) and (4 <= k <= 7) \
       and (th >= 0.10) and (ecoh <= 0.10) and (bokeh < 0.24) and (hf_ratio < 1.10) \
       and (straight_score < 0.10) and (dots < 1.1):
        comic += 0.24
    comic = max(0.0, comic)

    # Colored cartoon guard (tightened)
    if (0.16 <= s_mean <= 0.42) and (0.12 <= s_std <= 0.35) and (4 <= k <= 7) \
       and (th >= 0.10) and (ecoh <= 0.10) and (bokeh < 0.24) and (hf_ratio < 1.10) \
       and (straight_score < 0.10) and (dots < 1.1):
        comic += 0.24

    comic = max(0.0, comic)

    # Manga
    manga = 0.0
    if gray_ratio > 0.60 and ed > 0.032 and ecoh > 0.24 and hf_ratio < 1.08:
        manga += 0.60
    if dots > 1.0:
        manga += 0.15
    if s_mean > 0.15:
        manga -= 0.10
    manga = max(0.0, manga)

    # Children sketches
    child_sketch = 0.0
    if s_mean < 0.16:
        child_sketch += 0.20
    else:
        child_sketch -= 0.15
    if k <= 4:
        child_sketch += 0.15
    else:
        child_sketch -= 0.10
    if hf_ratio < 1.06:
        child_sketch += 0.15
    else:
        child_sketch -= 0.15
    if th < 0.016:
        child_sketch += 0.15
    else:
        child_sketch -= 0.10
    if ed > 0.030 and ecoh < 0.18:
        child_sketch += 0.20
    else:
        child_sketch -= 0.10
    if bokeh > 0.22 or (s_mean > 0.18 and s_std > 0.08):
        child_sketch -= 0.25
    child_sketch = max(0.0, child_sketch)

    # Classical oil painting
    oil = 0.0
    if brush_score > 0.46:
        oil += 0.30
    if hf_ratio < 1.26 and s_std > 0.06:
        oil += 0.18
    if contr > 0.12:
        oil += 0.10
    if sil < 0.07 and k >= 5:
        oil += 0.12
    if hf_ratio > 1.18:
        oil -= 0.14
    if bokeh > 0.30:
        oil -= 0.12
    if s_mean > 0.20 and s_std > 0.09:
        oil -= 0.08
    if k >= 7:
        oil -= 0.08
    oil = max(0.0, oil)

    # Science infographic / poster-like graphics
    sci_infog = 0.0
    if ecoh > 0.36:
        sci_infog += 0.30
    if sil > 0.12 and k <= 6:
        sci_infog += 0.20
    if straight_score > 0.28:
        sci_infog += 0.35
    if s_std < 0.09:
        sci_infog += 0.15

    poster_like = (
        (ecoh <= 0.06) and
        (th >= 0.10) and
        (hf_ratio < 0.90) and
        (k <= 4) and
        (sil >= 0.45) and
        (s_mean >= 0.32) and
        (s_std >= 0.18) and
        (straight_score < 0.12) and
        (dots < 0.6)
    )
    if poster_like:
        sci_infog += 0.28
        comic = max(0.0, comic - 0.12)

    sci_infog = max(0.0, sci_infog)

    # Watercolor
    watercolor = 0.0
    if s_mean < 0.22 and s_std < 0.07:
        watercolor += 0.30
    if th < 0.018 and ed < 0.04:
        watercolor += 0.25
    if hf_ratio < 1.12:
        watercolor += 0.20
    if sil < 0.08 and k <= 6:
        watercolor += 0.25
    watercolor = max(0.0, watercolor)

    # Technical drawing
    technical = 0.0
    if straight_score > 0.32:
        technical += 0.45
    if ecoh > 0.32:
        technical += 0.25
    if gray_ratio > 0.45 or s_std < 0.06:
        technical += 0.20
    if ed > 0.028 and th < 0.02:
        technical += 0.10
    technical = max(0.0, technical)

    # Scribble/sketches
    scribble = 0.0
    if s_mean < 0.14 and s_std < 0.055 and k <= 4:
        scribble += 0.22
    if ed > 0.052 and ecoh < 0.18 and th < 0.018:
        scribble += 0.20
    if hf_ratio < 1.06:
        scribble += 0.15
    if bokeh < 0.16:
        scribble += 0.08

    photo_like = int(hf_ratio > 1.15) + int(bokeh > 0.26) + int(s_mean > 0.20 and s_std > 0.09) + int(k >= 7)
    comic_like = int(ecoh >= 0.20) + int(0.012 < th < 0.060) + int(hf_ratio < 1.12) + int(bokeh < 0.22)
    if photo_like >= 2:
        scribble -= 0.30
    if comic_like >= 3 and s_mean >= 0.16:
        scribble -= 0.25
    scribble = max(0.0, scribble)

    scores = {
        "comic": float(min(1.0, comic)),
        "manga": float(min(1.0, manga)),
        "photo": float(min(1.0, photo)),
        "science illustration/infographic": float(min(1.0, sci_infog)),
        "classical oil painting": float(min(1.0, oil)),
        "watercolor": float(min(1.0, watercolor)),
        "children sketches": float(min(1.0, child_sketch)),
        "technical drawing/technical sketch": float(min(1.0, technical)),
        "scribble/sketches": float(min(1.0, scribble)),
    }
    primary = max(scores, key=scores.get)

    return StyleAnalysis(
        edge_density=ed,
        edge_coherence=ecoh,
        edge_thickness_score=th,
        saturation_mean=s_mean,
        saturation_std=s_std,
        color_clusters=k,
        color_silhouette=sil,
        contrast=contr,
        grayscale_ratio=gray_ratio,
        grain_score=grain,
        hf_ratio=hf_ratio,
        dot_pattern_score=dots,
        straight_line_score=straight_score,
        brush_texture_score=brush_score,
        bokeh_score=bokeh,
        class_scores=scores,
        primary_class=primary,
    )


def _base_descriptors_for_class(a: StyleAnalysis) -> List[str]:
    c = a.primary_class
    d: List[str] = []

    if c == "comic":
        d.append("clear line art" if a.edge_coherence < 0.34 or a.edge_thickness_score < 0.03 else "bold outlines")
        d.append("flat colors" if a.saturation_mean < 0.18 and a.saturation_std < 0.06 and a.color_clusters <= 5 else "balanced palette")
        if a.contrast > 0.16:
            d.append("high contrast")
        if a.dot_pattern_score > 0.9:
            d.append("screen-tone dots")
        if a.hf_ratio < 1.1 and a.grain_score < 70.0:
            d.append("clean finish")

    elif c == "manga":
        d += ["monochrome", "clear line art"]
        if a.dot_pattern_score > 0.8:
            d.append("halftone shading")
        if a.contrast > 0.15:
            d.append("high contrast")
        if a.hf_ratio < 1.15:
            d.append("clean finish")

    elif c == "photo":
        d.append("natural lighting")
        if a.hf_ratio > 1.20:
            d.append("fine detail")
        d.append("smooth gradients")
        if a.saturation_std > 0.1:
            d.append("rich colors")
        if a.bokeh_score > 0.30:
            d.append("shallow depth of field")

    elif c == "science illustration/infographic":
        d += ["thin precise lines", "flat colors", "clean layout"]
        if a.straight_line_score > 0.35:
            d.append("geometric accuracy")
        if a.saturation_std < 0.08:
            d.append("limited palette")

    elif c == "classical oil painting":
        d += ["brush textures", "rich tones", "soft transitions"]
        if a.brush_texture_score > 0.45:
            d.append("impasto strokes")

    elif c == "watercolor":
        d += ["soft washes", "bleeding edges", "delicate tones"]
        if a.saturation_mean < 0.2:
            d.append("pastel palette")

    elif c == "children sketches":
        d += ["simple shapes", "thin uneven lines", "playful composition"]

    elif c == "technical drawing/technical sketch":
        d += ["precise straight lines", "monochrome", "high clarity"]

    else:
        d += ["loose strokes", "expressive lines", "dynamic texture"]

    out: List[str] = []
    for x in d:
        if x not in out:
            out.append(x)
    return out[:5]


def extract_style_descriptors(image_path: Path, debug: bool = False) -> List[str]:
    a = analyze_style(image_path)
    desc = _base_descriptors_for_class(a)
    if debug:
        print("[style:debug] primary_class:", a.primary_class)
        print("[style:debug] class_scores:", a.class_scores)
        print(
            "[style:debug] metrics:",
            f"ed={a.edge_density:.3f}",
            f"ecoh={a.edge_coherence:.3f}",
            f"th={a.edge_thickness_score:.3f}",
            f"s_mean={a.saturation_mean:.3f}",
            f"s_std={a.saturation_std:.3f}",
            f"k={a.color_clusters}",
            f"sil={a.color_silhouette:.3f}",
            f"contrast={a.contrast:.3f}",
            f"gray_ratio={a.grayscale_ratio:.3f}",
            f"grain={a.grain_score:.1f}",
            f"hf_ratio={a.hf_ratio:.2f}",
            f"dots={a.dot_pattern_score:.2f}",
            f"straight={a.straight_line_score:.2f}",
            f"brush={a.brush_texture_score:.2f}",
            f"bokeh={a.bokeh_score:.2f}",
        )
        print("[style:debug] descriptors:", desc)
    return desc

def detect_primary_style_label(image_path: Path) -> str:
    return analyze_style(image_path).primary_class

def extract_style_with_label(image_path: Path, debug: bool = False) -> Tuple[str, List[str]]:
    a = analyze_style(image_path)
    desc = _base_descriptors_for_class(a)
    if debug:
        print("[style:debug] primary_class:", a.primary_class)
        print("[style:debug] class_scores:", a.class_scores)
        print("[style:debug] descriptors:", desc)
    return a.primary_class, desc
