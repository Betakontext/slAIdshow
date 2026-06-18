# image_backend_patch_optional_ipadapter.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Drop-in utility for LocalComfyBackend to inject a style reference image
# into IP-Adapter nodes when available in the workflow. This keeps the
# bridge contract unchanged (send prompt_dict as-is to ComfyUI).

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

from style_engine import StyleConfig, ReferenceStore, apply_ipadapter_if_present

def maybe_apply_reference(
    prompt_dict: Dict[str, Any],
    references_dir: Path,
    style_cfg: StyleConfig,
) -> Dict[str, Any]:
    """If style usage is enabled and id resolves, patch IP-Adapter nodes; otherwise no-op."""
    if not style_cfg.use_reference or not style_cfg.reference_id:
        return prompt_dict

    store = ReferenceStore(references_dir)
    ref_path = store.get_path(style_cfg.reference_id)
    if ref_path is None:
        # Reference missing; silently keep text-only styling
        return prompt_dict

    try:
        apply_ipadapter_if_present(prompt_dict, image_path=ref_path, strength=style_cfg.reference_strength)
        return prompt_dict
    except Exception:
        # Never fail generation due to style reference glitches
        return prompt_dict
