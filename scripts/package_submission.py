"""Package a trained LoRA adapter into submission.zip for Kaggle."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from nemotron.config import OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Package LoRA adapter as submission.zip")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=OUTPUT_DIR / "sft_output" / "lora_adapter",
        help="Directory containing adapter_model.safetensors and adapter_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "submission.zip",
        help="Output zip path",
    )
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in os.listdir(args.adapter_dir):
            filepath = args.adapter_dir / filename
            if filepath.is_file():
                zf.write(filepath, filename)

    zip_size = args.output.stat().st_size
    print(f"Created {args.output} ({zip_size / 1024 / 1024:.2f} MB)")

    with zipfile.ZipFile(args.output, "r") as zf:
        print("Contents:")
        for info in zf.infolist():
            print(f"  {info.filename}: {info.file_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
