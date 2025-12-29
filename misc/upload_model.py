#!/usr/bin/env python
"""
Upload all subfolders inside `weights/` to Hugging Face Hub.

For each subfolder, create (or reuse) a model repo:
  2025FDU-CG/<subfolder_name>

and upload all files in that subfolder.

Usage example:
  HF_TOKEN=your_token_here python upload_all_weights_to_hf.py \
      --weights_dir ./weights \
      --org 2025FDU-CG
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./weights",
        help="Local directory that contains subfolders of checkpoints.",
    )
    parser.add_argument(
        "--org",
        type=str,
        default="2025FDU-CG",
        help="Hugging Face organization or username to upload to.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token. "
             "If not provided, will use HF_TOKEN env variable.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private (default is public).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print what would be uploaded, without actually uploading.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if token is None:
        raise ValueError(
            "No token found. Please either set --token or export HF_TOKEN env variable."
        )

    api = HfApi(token=token)

    weights_dir = os.path.abspath(args.weights_dir)
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    subfolders = [
        d for d in sorted(os.listdir(weights_dir))
        if os.path.isdir(os.path.join(weights_dir, d))
    ]

    if not subfolders:
        print(f"No subfolders found under {weights_dir}")
        return

    print(f"Found {len(subfolders)} subfolders under {weights_dir}:")
    for name in subfolders:
        print(f"  - {name}")

    for name in subfolders:
        local_path = os.path.join(weights_dir, name)
        repo_id = f"{args.org}/{name}"

        print("\n" + "=" * 60)
        print(f"Processing subfolder: {local_path}")
        print(f"Target HF repo: {repo_id}")

        if args.dry_run:
            print("[DRY RUN] Would create repo and upload this folder.")
            continue

        # 1) Create repo if needed

        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            private=args.private,
            exist_ok=True,  # do not crash if it already exists
        )
        print(f"Repo ready: {repo_id}")

        # 2) Upload the entire folder
        upload_folder(
            repo_id=repo_id,
            token=token,
            repo_type="model",
            folder_path=local_path,
        )
        print(f"Uploaded folder {local_path} to {repo_id}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
