#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Testskript für comfyui.py
- Prüft ComfyUI-Verfügbarkeit auf localhost
- Lädt optional ein exportiertes ComfyUI-Workflow-JSON und überschreibt dynamische Felder
- Startet den Job, pollt /history und lädt die erzeugten Bilder herunter
- Speichert in APP_OUTPUT_DIR (Default: ./outputs/images)

Voraussetzungen:
- ComfyUI läuft lokal auf 127.0.0.1:8188
- APP_DISABLE_COMFYUI=0 gesetzt (sonst bricht comfyui.py ab)
- comfyui.py liegt im gleichen Verzeichnis oder ist importierbar (PYTHONPATH anpassen)

Beispiele:
  APP_DISABLE_COMFYUI=0 python test_comfy_local.py --prompt "Ein ruhiger See bei Sonnenuntergang" --width 512 --height 512 --steps 20
  APP_DISABLE_COMFYUI=0 python test_comfy_local.py --workflow ./workflows/text2img.json --prompt "Illustration eines Roboters im Klassenzimmer" --seed 1234
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Import des lokalen comfyui-Moduls
try:
    import comfyui  # type: ignore
except Exception as e:
    print(f"[ERR] Konnte 'comfyui' nicht importieren: {e}")
    print("       Stelle sicher, dass comfyui.py im selben Ordner liegt, oder setze PYTHONPATH korrekt.")
    sys.exit(2)


def _env_str(k: str, d: str) -> str:
    return (os.getenv(k, d) or "").strip()


def _coalesce_int(a: Optional[int], b: Optional[int], default: int) -> int:
    for v in (a, b, default):
        if v is not None:
            return int(v)
    return default


def _coalesce_float(a: Optional[float], b: Optional[float], default: float) -> float:
    for v in (a, b, default):
        if v is not None:
            return float(v)
    return default


def _load_workflow(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Workflow-Datei nicht gefunden: {path}")
    with path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            raise ValueError(f"Ungültiges Workflow-JSON ({path}): {e}")
    if not isinstance(data, dict):
        raise ValueError("Workflow-JSON hat kein Dict als Wurzel.")
    return data


def _try_patch_workflow(
    wf: Dict[str, Any],
    prompt: str,
    *,
    negative_prompt: Optional[str],
    seed: Optional[int],
    width: Optional[int],
    height: Optional[int],
    steps: Optional[int],
    cfg: Optional[float],
    sampler_name: Optional[str],
) -> Dict[str, Any]:
    """
    Versucht, die üblichen Felder in einem exportierten ComfyUI-Workflow zu überschreiben.
    Diese Funktion ist defensiv: existieren Knoten/Felder nicht, wird leise übersprungen.
    """
    # Häufiger Aufbau: Root-Key "prompt" mit Node-IDs als Strings
    root = wf.get("prompt", wf)
    if not isinstance(root, dict):
        root = wf

    # Hilfsfunktion: Finde alle Nodes eines Typs
    def nodes_of(class_type: str):
        for node_id, desc in root.items():
            if isinstance(desc, dict) and desc.get("class_type") == class_type:
                yield node_id, desc

    # Positive Prompt (CLIPTextEncode)
    for node_id, desc in nodes_of("CLIPTextEncode"):
        try:
            inputs = desc.get("inputs", {})
            if isinstance(inputs, dict) and "text" in inputs:
                inputs["text"] = prompt
                print(f"[PATCH] CLIPTextEncode@{node_id}.inputs.text gesetzt.")
                break
        except Exception:
            pass

    # Negative Prompt (zweiter CLIPTextEncode oder expliziter negativer Encoder)
    if negative_prompt:
        patched = False
        for node_id, desc in nodes_of("CLIPTextEncode"):
            try:
                inputs = desc.get("inputs", {})
                if isinstance(inputs, dict) and "text" in inputs:
                    # Heuristik: Negative Prompt in den zweiten Encode schreiben, wenn es >1 gibt
                    inputs["text"] = negative_prompt
                    print(f"[PATCH] (neg) CLIPTextEncode@{node_id}.inputs.text gesetzt.")
                    patched = True
                    break
            except Exception:
                pass
        if not patched:
            print("[PATCH] Hinweis: Kein separater negativer Prompt-Knoten erkannt; Überschreibung übersprungen.")

    # Seed, Steps, CFG, Sampler am KSampler
    for node_id, desc in nodes_of("KSampler"):
        try:
            inputs = desc.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            if seed is not None:
                inputs["seed"] = int(seed)
                print(f"[PATCH] KSampler@{node_id}.inputs.seed={seed}")
            if steps is not None:
                inputs["steps"] = int(steps)
                print(f"[PATCH] KSampler@{node_id}.inputs.steps={steps}")
            if cfg is not None:
                inputs["cfg"] = float(cfg)
                print(f"[PATCH] KSampler@{node_id}.inputs.cfg={cfg}")
            if sampler_name:
                inputs["sampler_name"] = sampler_name
                print(f"[PATCH] KSampler@{node_id}.inputs.sampler_name={sampler_name}")
        except Exception:
            pass

    # Größe am EmptyLatentImage
    for node_id, desc in nodes_of("EmptyLatentImage"):
        try:
            inputs = desc.get("inputs", {})
            if not isinstance(inputs, dict):
                continue
            if width is not None:
                inputs["width"] = int(width)
                print(f"[PATCH] EmptyLatentImage@{node_id}.inputs.width={width}")
            if height is not None:
                inputs["height"] = int(height)
                print(f"[PATCH] EmptyLatentImage@{node_id}.inputs.height={height}")
            if "batch_size" in inputs and not inputs.get("batch_size"):
                inputs["batch_size"] = 1
        except Exception:
            pass

    return wf


def main() -> int:
    parser = argparse.ArgumentParser(description="Lokaler Test für comfyui.py gegen ComfyUI auf localhost.")
    parser.add_argument("--workflow", type=str, default="", help="Pfad zu einem exportierten ComfyUI-Workflow-JSON")
    parser.add_argument("--prompt", type=str, required=False, default="A serene landscape lake at sunset, golden hour, high detail, photorealistic",
                        help="Bildbeschreibung (Prompt)")
    parser.add_argument("--neg", type=str, default="text, watermark, logo, low quality, bad anatomy, blurry", help="Negativer Prompt")
    parser.add_argument("--seed", type=int, default=None, help="Seed (optional)")
    parser.add_argument("--width", type=int, default=512, help="Bildbreite")
    parser.add_argument("--height", type=int, default=512, help="Bildhöhe")
    parser.add_argument("--steps", type=int, default=20, help="Sampling-Schritte")
    parser.add_argument("--cfg", type=float, default=6.0, help="CFG Scale")
    parser.add_argument("--sampler", type=str, default="dpmpp_2m", help="Sampler-Name (z. B. euler, dpmpp_2m)")
    parser.add_argument("--timeout", type=float, default=300.0, help="Max. Wartezeit in Sekunden (History Polling)")
    args = parser.parse_args()

    # ENV-Sanity: APP_DISABLE_COMFYUI
    disabled = (os.getenv("APP_DISABLE_COMFYUI", "1") or "1").strip()
    if disabled in {"1", "true", "yes", "on"}:
        print("[ERR] APP_DISABLE_COMFYUI=1 → ComfyUI ist deaktiviert. Setze APP_DISABLE_COMFYUI=0 und starte erneut.")
        return 3

    # Verfügbarkeit prüfen
    import asyncio
    ok = asyncio.run(comfyui.comfy_available())
    if not ok:
        print("[ERR] ComfyUI nicht erreichbar auf 127.0.0.1:8188. Starte ComfyUI und versuche es erneut.")
        return 4

    # Workflow laden/erstellen
    wf: Optional[Dict[str, Any]] = None
    if args.workflow:
        try:
            wf = _load_workflow(Path(args.workflow))
            wf = _try_patch_workflow(
                wf,
                prompt=args.prompt,
                negative_prompt=args.neg or None,
                seed=args.seed,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg=args.cfg,
                sampler_name=args.sampler,
            )
        except Exception as e:
            print(f"[ERR] Workflow konnte nicht geladen/gepatcht werden: {e}")
            return 5
    else:
        # Fallback: Verwende den Default-Workflow aus comfyui.py (benötigt passendes SDXL-Checkpoint!)
        print("[INFO] Kein Workflow-JSON angegeben. Verwende build_default_text2img_workflow() aus comfyui.py.")
        wf = comfyui.build_default_text2img_workflow(
            prompt=args.prompt,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg=args.cfg,
            sampler_name=args.sampler,
        )

    # Job starten
    print("[RUN] Sende Prompt an ComfyUI…")
    try:
        pid = asyncio.run(comfyui.post_prompt(wf))
        print(f"[RUN] prompt_id = {pid}")
    except Exception as e:
        print(f"[ERR] /prompt fehlgeschlagen: {e}")
        return 6

    # History pollen
    try:
        print("[RUN] Pollen von /history …")
        hist = asyncio.run(comfyui.poll_history_until_done(pid, poll_interval=1.0, max_wait_sec=float(args.timeout)))
    except Exception as e:
        print(f"[ERR] Polling fehlgeschlagen/Timeout: {e}")
        return 7

    # Downloads
    try:
        out_dir = Path(_env_str("APP_OUTPUT_DIR", "./outputs/images")).resolve()
        paths = asyncio.run(comfyui.download_outputs(hist, out_dir))
        print("[OK] Bilder gespeichert:")
        for p in paths:
            print(f" - {p}")
        return 0
    except Exception as e:
        print(f"[ERR] Download der Outputs fehlgeschlagen: {e}")
        return 8


if __name__ == "__main__":
    raise SystemExit(main())
