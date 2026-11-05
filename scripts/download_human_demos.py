"""Download the human demonstration videos used for evaluation.

Usage:
    python scripts/download_human_demos.py
    python scripts/download_human_demos.py --distraction no --tasks open_lid --cameras front
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from huggingface_hub import snapshot_download

from benchmark import HUMAN_DEMO_CAMERAS, HUMAN_DEMO_IDS, TASK_NAMES, human_track_name

REPO_ID = "LogosRoboticsLab/roboreel"
REMOTE_ROOT = "human_demos/eval"


def parse_args():
    p = argparse.ArgumentParser(description="Download the human demo videos used for evaluation")
    p.add_argument("--distraction", choices=["no", "with", "both"], default="both")
    p.add_argument("--tasks", nargs="+", choices=sorted(TASK_NAMES.values()), default=sorted(TASK_NAMES.values()))
    p.add_argument("--cameras", nargs="+", choices=HUMAN_DEMO_CAMERAS, default=list(HUMAN_DEMO_CAMERAS))
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent.parent),
        help="Videos are saved in <output-dir>/human_demos/eval (default: the repository)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    tracks = {
        "no": [human_track_name(False)],
        "with": [human_track_name(True)],
        "both": [human_track_name(False), human_track_name(True)],
    }[args.distraction]
    files = [
        f"{REMOTE_ROOT}/{track}/{task}/{demo_id:04d}/{camera}.mp4"
        for track in tracks
        for task in args.tasks
        for demo_id in HUMAN_DEMO_IDS
        for camera in args.cameras
    ]

    print(f"Downloading {len(files)} videos from {REPO_ID}...")
    snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=files, local_dir=args.output_dir)

    # A pattern that matches nothing isn't an error for snapshot_download.
    output_dir = Path(args.output_dir)
    missing = [file for file in files if not (output_dir / file).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(files)} videos weren't found in {REPO_ID}. "
            f"First missing: {missing[0]}"
        )
    print(f"Saved {len(files)} videos in {output_dir / REMOTE_ROOT}")


if __name__ == "__main__":
    main()
