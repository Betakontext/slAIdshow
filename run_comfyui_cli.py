# run_comfyui_cli.py – Update: KeyboardInterrupt sauber behandeln
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List
import importlib

def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyUI CLI test runner")
    parser.add_argument("--workflow", required=True, help="Pfad zur LiteGraph JSON-Datei (oder API-Export)")
    parser.add_argument("--out", default="./outputs", help="Ausgabeordner")
    parser.add_argument("--prompt", required=True, help="Prompt-Text")
    parser.add_argument("--negative", default="text, watermark, logo, low quality, blurry, bad anatomy")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--sampler", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--denoise", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--comfy_output_dir", type=str, default="")
    parser.add_argument("--max_wait", type=float, default=180.0)
    parser.add_argument("--poll", type=float, default=1.0)
    args = parser.parse_args()

    os.environ.setdefault("APP_COMFY_HOST", "127.0.0.1")
    os.environ.setdefault("APP_COMFY_PORT", "8188")
    os.environ.setdefault("APP_COMFY_HTTP_TIMEOUT", "45")

    cb = importlib.import_module("comfyui_bridge")

    async def run() -> int:
        print("[CLI] Using comfyui_bridge from:", cb.__file__)
        wf_path = Path(args.workflow).expanduser().resolve()
        out_dir = Path(args.out).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        copy_from = Path(args.comfy_output_dir).expanduser().resolve() if args.comfy_output_dir else None
        if copy_from and not copy_from.exists():
            print(f"[CLI][warn] --comfy_output_dir not found: {copy_from} (Fallback ignoriert)")

        print(f"[CLI] Workflow: {wf_path}")
        print(f"[CLI] Output dir: {out_dir}")
        if copy_from:
            print(f"[CLI] Fallback source: {copy_from}")

        try:
            images: List[Path] = await cb.generate_from_litegraph_file(
                litegraph_path=wf_path,
                out_dir=out_dir,
                prompt_text=args.prompt,
                negative_text=args.negative,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfg=args.cfg,
                sampler_name=args.sampler,
                scheduler=args.scheduler,
                denoise=args.denoise,
                seed=args.seed,
                max_wait_sec=args.max_wait,
                poll_interval=args.poll,
                copy_from_comfy_dir=copy_from,
            )
            print("[CLI] Generated images:")
            for p in images:
                print(" -", p)
            return 0
        except KeyboardInterrupt:
            print("\n[CLI] Abbruch per KeyboardInterrupt.")
            return 130  # übliche Exit-Codes für SIGINT
        except Exception as e:
            import traceback
            print("[CLI][ERROR]", repr(e))
            traceback.print_exc()
            dbg1 = out_dir / "debug_last_history.json"
            dbg2 = out_dir / "debug_api_prompt.json"
            if dbg1.exists() or dbg2.exists():
                print("[CLI][hint] Debug-Dumps im Output-Ordner vorhanden.")
            return 1

    try:
        exit_code = asyncio.run(run())
    except KeyboardInterrupt:
        # Falls der Abbruch den Event-Loop trifft
        print("\n[CLI] Abbruch per KeyboardInterrupt (outer).")
        exit_code = 130
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
