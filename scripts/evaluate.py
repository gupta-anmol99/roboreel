"""Evaluate a policy on the benchmark.

Usage:
    python scripts/evaluate.py --env EmptyBasket-v1 \
        --policy examples.random_policy:RandomPolicy

Use --env all to run every task, or list several task IDs.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path.cwd()))

from benchmark import (
    BENCHMARK_VERSION,
    CONTROL_MODES,
    EPISODES_PER_RUN,
    INIT_STATES_DIR_ENV_VAR,
    TASK_NAMES,
    HUMAN_DEMOS_DIR_ENV_VAR,
    evaluate,
    results_track_name,
)


def load_policy(spec: str, kwargs: dict):
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"--policy must look like module.path:ClassName, got '{spec}'")
    return getattr(importlib.import_module(module_name), class_name)(**kwargs)


def resolve_env_ids(names: list[str]) -> list[str]:
    if names == ["all"]:
        return list(TASK_NAMES)
    unknown = [name for name in names if name not in TASK_NAMES]
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}. Choose from {list(TASK_NAMES)} or use 'all'.")
    return names


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a policy from fixed initial states")
    p.add_argument("--env", type=str, nargs="+", required=True, help="Task IDs, or 'all'")
    p.add_argument("--policy", type=str, required=True, help="module.path:ClassName")
    p.add_argument("--policy-kwargs", type=json.loads, default={}, help="JSON constructor arguments")
    p.add_argument(
        "--init-states-dir",
        type=str,
        default=None,
        help=(
            "Folder holding with_distraction/<task>.pkl and without_distraction/<task>.pkl "
            f"(default: ${INIT_STATES_DIR_ENV_VAR}, then the repository's assets/data/init_states/)"
        ),
    )
    p.add_argument("--distract", action="store_true", help="Evaluate on the scenes with distractors")
    p.add_argument(
        "--human-distract", action="store_true", help="Use the human videos with distractors"
    )
    p.add_argument(
        "--human-demos-dir",
        type=str,
        default=None,
        help=f"Folder holding eval/ (default: ${HUMAN_DEMOS_DIR_ENV_VAR}, then the repository's human_demos/)",
    )
    episodes = p.add_mutually_exclusive_group()
    episodes.add_argument(
        "--itr",
        type=int,
        default=1,
        help=f"Number of runs of {EPISODES_PER_RUN} episodes: 1 runs episodes 0-19, 2 also runs 20-39, and so on",
    )
    episodes.add_argument("--episodes", type=int, nargs="+", default=None, help="Run only these episodes, as one run")
    p.add_argument("--obs-mode", type=str, default="rgb")
    p.add_argument("--control-mode", type=str, default="pd_joint_pos", choices=CONTROL_MODES)
    p.add_argument("--max-steps", type=int, default=None, help="Default: the benchmark's step limit")
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument("--save-video", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    env_ids = resolve_env_ids(args.env)
    policy = load_policy(args.policy, args.policy_kwargs)

    success_rates = {}
    success_rate_stds = {}
    for env_id in env_ids:
        results = evaluate(
            policy,
            env_id=env_id,
            init_states_dir=args.init_states_dir,
            distract=args.distract,
            human_distract=args.human_distract,
            human_demos_dir=args.human_demos_dir,
            itr=args.itr,
            episode_ids=args.episodes,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            save_video=args.save_video,
        )
        for run in results["runs"]:
            print(f"{env_id} run {run['run']}: {run['success_rate']:.3f} ({len(run['episodes'])} episodes)")
        print(
            f"{env_id} success rate: {results['success_rate']:.3f} "
            f"+/- {results['success_rate_std']:.3f} ({results['num_runs']} runs)"
        )
        success_rates[env_id] = results["success_rate"]
        success_rate_stds[env_id] = results["success_rate_std"]

    if len(env_ids) > 1:
        mean_success_rate = sum(success_rates.values()) / len(success_rates)
        print(f"Mean success rate: {mean_success_rate:.3f} ({len(env_ids)} tasks)")
        track = results_track_name(args.distract, args.human_distract)
        with open(Path(args.output_dir) / track / "summary.json", "w") as f:
            json.dump(
                {
                    "benchmark_version": BENCHMARK_VERSION,
                    "distract": args.distract,
                    "human_distract": args.human_distract,
                    "obs_mode": args.obs_mode,
                    "control_mode": args.control_mode,
                    "mean_success_rate": mean_success_rate,
                    "success_rates": success_rates,
                    "success_rate_stds": success_rate_stds,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
