#!/usr/bin/env python3
"""
ljspeech_to_jsonl.py
====================
Convert an LJSpeech-style dataset to the OmniVoice JSONL manifest format.

LJSpeech metadata.csv format (pipe-separated, NO header row):
    <file_stem>|<raw_transcription>|<normalized_transcription>

    e.g.
    LJ001-0001|Printing, in the only sense ...|Printing, in the only sense ...
    LJ001-0002|in being comparatively modern|in being comparatively modern

The audio files live in <dataset_root>/wavs/<file_stem>.wav

OmniVoice JSONL output (one JSON object per line):
    {"id": "LJ001-0001", "audio_path": "/abs/path/to/wavs/LJ001-0001.wav",
     "text": "Printing, in the only sense ...", "language_id": "en"}

Usage
-----
    python examples/ljspeech_to_jsonl.py \\
        --dataset_dir  /path/to/LJSpeech-1.1 \\
        --output_dir   data/ljspeech \\
        --dev_count    100 \\
        --seed         42 \\
        --language_id  en \\
        --use_normalized

Arguments
---------
    --dataset_dir     Root of the LJSpeech dataset (contains metadata.csv and wavs/).
    --output_dir      Directory where train.jsonl and dev.jsonl will be written.
    --dev_count       Number of samples to hold out for the dev set (default: 100).
                      Set to 0 or use --no_split to write a single train.jsonl.
    --seed            Random seed for the train/dev split (default: 42).
    --language_id     language_id field in the output JSONL (default: en).
    --use_normalized  If set, use the 3rd column (normalised text); otherwise use
                      the 2nd column (raw transcription). Default: use normalized.
    --audio_ext       Audio file extension to look for in wavs/ (default: wav).
    --no_split        If set, write a single output.jsonl instead of splitting.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_metadata(metadata_path: Path, use_normalized: bool, audio_ext: str, wavs_dir: Path):
    """
    Parse LJSpeech metadata.csv and return a list of dicts ready for JSONL.

    The CSV uses '|' as the delimiter and has NO header line.
    Column layout:
        0 – file stem  (e.g. LJ001-0001)
        1 – raw transcription
        2 – normalised transcription  (may be missing on some forks)
    """
    records = []
    missing_audio = []

    with metadata_path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n\r")
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 2:
                print(f"  [WARN] Line {lineno}: expected at least 2 columns, got {len(parts)} – skipping.",
                      file=sys.stderr)
                continue

            stem = parts[0].strip()

            # Column selection
            if use_normalized and len(parts) >= 3 and parts[2].strip():
                text = parts[2].strip()
            else:
                text = parts[1].strip()

            # Resolve absolute audio path
            audio_path = wavs_dir / stem
            if not audio_path.exists():
                missing_audio.append(str(audio_path))
                continue

            records.append({
                "id": stem,
                "audio_path": str(audio_path.resolve()),
                "text": text,
            })

    if missing_audio:
        print(f"\n  [WARN] {len(missing_audio)} audio file(s) not found and skipped:", file=sys.stderr)
        for p in missing_audio[:10]:
            print(f"         {p}", file=sys.stderr)
        if len(missing_audio) > 10:
            print(f"         … and {len(missing_audio) - 10} more.", file=sys.stderr)

    return records


def write_jsonl(records: list, path: Path, language_id: str):
    """Write records to a JSONL file, adding language_id to each entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            obj = {
                "id": rec["id"],
                "audio_path": rec["audio_path"],
                "text": rec["text"],
                "language_id": language_id,
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  Written {len(records):>6,} samples → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert an LJSpeech dataset to OmniVoice JSONL manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir", required=True, type=Path,
        help="Root directory of the LJSpeech dataset (must contain metadata.csv and wavs/).",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Directory where train.jsonl (and optionally dev.jsonl) will be written.",
    )
    parser.add_argument(
        "--dev_count", type=int, default=100,
        help="Number of samples to hold out for the dev set (default: 100). "
             "Set to 0 or use --no_split to output a single train.jsonl instead.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used for the train/dev split.",
    )
    parser.add_argument(
        "--language_id", type=str, default="zh",
        help="language_id to embed in every JSONL record.",
    )
    parser.add_argument(
        "--use_normalized", action="store_true", 
        help="Use the normalised transcription column (column 3) when available. "
             "Pass --no_use_normalized to use the raw column instead.",
    )
    parser.add_argument(
        "--no_use_normalized", dest="use_normalized", action="store_false",
        help="Use the raw transcription column (column 2) instead of the normalised one.",
    )
    parser.add_argument(
        "--audio_ext", type=str, default="wav",
        help="Extension of audio files inside the wavs/ directory.",
    )
    parser.add_argument(
        "--no_split", action="store_true",
        help="Write a single output.jsonl instead of separate train/dev files.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    dataset_dir: Path = args.dataset_dir.resolve()
    metadata_path = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"

    if not dataset_dir.is_dir():
        parser.error(f"--dataset_dir does not exist: {dataset_dir}")
    if not metadata_path.is_file():
        parser.error(f"metadata.csv not found in: {dataset_dir}")
    if not wavs_dir.is_dir():
        parser.error(f"wavs/ directory not found in: {dataset_dir}")

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------
    print(f"\nLJSpeech → OmniVoice JSONL converter")
    print(f"  dataset_dir  : {dataset_dir}")
    print(f"  metadata.csv : {metadata_path}")
    print(f"  wavs/        : {wavs_dir}")
    print(f"  audio_ext    : {args.audio_ext}")
    print(f"  use_normalized: {args.use_normalized}")
    print(f"  language_id  : {args.language_id}")
    print()

    records = parse_metadata(metadata_path, args.use_normalized, args.audio_ext, wavs_dir)

    if not records:
        print("[ERROR] No valid records found. Check your dataset_dir and metadata.csv.", file=sys.stderr)
        sys.exit(1)

    print(f"  Parsed {len(records):,} valid samples from metadata.csv.")

    # ------------------------------------------------------------------
    # Split / write
    # ------------------------------------------------------------------
    output_dir: Path = args.output_dir.resolve()

    if args.no_split or args.dev_count <= 0:
        out_path = output_dir / "train.jsonl"
        write_jsonl(records, out_path, args.language_id)
        print("\nDone.")
        return

    n_dev = args.dev_count
    if n_dev >= len(records):
        parser.error(f"--dev_count ({n_dev}) must be less than total samples ({len(records)}).")

    # Reproducible shuffle then split
    rng = random.Random(args.seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    dev_records   = shuffled[:n_dev]
    train_records = shuffled[n_dev:]

    print(f"\n  Split (seed={args.seed}, dev_count={n_dev}):")
    print(f"    train: {len(train_records):,} samples")
    print(f"    dev  : {len(dev_records):,} samples")
    print()

    write_jsonl(train_records, output_dir / "train.jsonl", args.language_id)
    write_jsonl(dev_records,   output_dir / "dev.jsonl",   args.language_id)

    print(f"\nDone. Files written to: {output_dir}")
    print("\nNext step — tokenise audio:")
    print(f"  python -m omnivoice.scripts.extract_audio_tokens \\")
    print(f"      --input_jsonl {output_dir / 'train.jsonl'} \\")
    print(f"      --tar_output_pattern  data/ljspeech/tokens/train/audios/shard-%06d.tar \\")
    print(f"      --jsonl_output_pattern data/ljspeech/tokens/train/txts/shard-%06d.jsonl")


if __name__ == "__main__":
    main()
