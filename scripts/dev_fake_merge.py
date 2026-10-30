#!/usr/bin/env python3
"""GPU-free stand-in for ``scripts/merge_lora.py``.

Writes small placeholder artifacts under ``--output_dir`` so the cockpit's
merge lifecycle can be exercised without CUDA or a real checkpoint. The files
satisfy the merge-completion checks (``config.json`` + ``model.safetensors``)
but contain no model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-free fake LoRA merge for the stateful cockpit"
    )
    parser.add_argument("--adapter_path", required=True, help="LoRA adapter dir")
    parser.add_argument("--output_dir", required=True, help="Merged model dir")
    # scripts/merge_lora.py also accepts --base_model/--device; tolerate any
    # extra real-merge flags without requiring them.
    args, _unknown = parser.parse_known_args(argv)
    return args


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_empty_safetensors(path: Path) -> None:
    """Write a valid safetensors container with an empty tensor header."""
    header = b"{}"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adapter_path = Path(args.adapter_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print(" Fake LoRA merge (GPU-free cockpit development)")
    print(f" Adapter Path: {adapter_path}")
    print(f" Output Dir:   {output_dir}")
    print("=======================================================", flush=True)

    if not (adapter_path / "adapter_config.json").is_file():
        # Placeholder merge intentionally tolerates a missing adapter so tests
        # and dev flows can exercise merge cleanup independently.
        print(f" warning: no adapter_config.json at {adapter_path}", flush=True)

    _write_json(
        output_dir / "config.json",
        {
            "model_type": "whisper",
            "architectures": ["WhisperForConditionalGeneration"],
            "base_model_name_or_path": "BuzzASR/persian",
            "adapter_path": str(adapter_path),
            "fake": True,
            "merged_with": "scripts/dev_fake_merge.py",
        },
    )
    _write_empty_safetensors(output_dir / "model.safetensors")
    _write_json(
        output_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "WhisperTokenizer",
            "processor_class": "WhisperProcessor",
            "fake": True,
        },
    )
    print(f"Fake merge complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
