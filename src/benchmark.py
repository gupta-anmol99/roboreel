"""RoboReel benchmark: a policy watches a human video, then performs the task in simulation.

Every episode starts from a saved scene in <init_states_dir>/<with_distraction or without_distraction>/<task>.pkl.
Bring any algorithm by implementing Policy; evaluate() handles everything else.
"""

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

import envs  # noqa: F401

INIT_STATES_DIR_ENV_VAR = "ROBOREEL_INIT_STATES_DIR"
DEFAULT_INIT_STATES_DIR = Path(__file__).parent.parent / "assets" / "data" / "init_states"
HUMAN_DEMOS_DIR_ENV_VAR = "ROBOREEL_HUMAN_DEMOS_DIR"
DEFAULT_HUMAN_DEMOS_DIR = Path(__file__).parent.parent / "human_demos"
ROBOT_UID = "panda_wristcam_hd"

BENCHMARK_VERSION = "v2"
CONTROL_MODES = ("pd_joint_pos", "pd_ee_delta_pose")

# Steps allowed per episode, for every task.
MAX_EPISODE_STEPS = 1000

# Each task's name in the init state files and the human demo folders.
TASK_NAMES = {
    "PickCube-v2": "pick_cube",
    "PushCube-v2": "push_cube",
    "CloseDrawer-v1": "close_drawer",
    "Toaster-v1": "press_toaster",
    "BowlOnPlate-v1": "bowl_in_plate",
    "OpenLid-v1": "open_lid",
    "PutBasket-v1": "toys_in_basket",
    "EmptyBasket-v1": "empty_basket",
    "StackCups-v1": "stack_cups",
    "ObjectInDrawer-v1": "toys_in_drawer",
}

EPISODES_PER_RUN = 20

# Every run uses these human demos, in order: a run's first scene gets 0080, its second 0081, and so on.
HUMAN_DEMO_IDS = range(80, 100)
HUMAN_DEMO_CAMERAS = ("egocentric", "front", "left", "right")


def track_name(distract: bool) -> str:
    return "with_distraction" if distract else "without_distraction"


def human_track_name(human_distract: bool) -> str:
    return "with_distraction" if human_distract else "no_distraction"


def results_track_name(distract: bool, human_distract: bool) -> str:
    return f"human_{human_track_name(human_distract)}__env_{track_name(distract)}"


@dataclass(frozen=True)
class Episode:
    env_id: str
    episode_id: int
    seed: int
    state: dict
    distractor_names: tuple[str, ...] = ()


class Policy:
    """Implement this to evaluate your algorithm."""

    def reset(self, episode_id: int, env: gym.Env, human_demo: dict[str, Path]) -> None:
        """Called once before each episode, e.g. to clear memory and load the task video.

        human_demo maps each downloaded camera ("front", ...) to its mp4 file.
        """

    def act(self, obs: Any) -> np.ndarray:
        """Return one action for one observation (nested dict of numpy arrays)."""
        raise NotImplementedError


def resolve_init_states_dir(init_states_dir: str | Path | None = None) -> Path:
    """Use the given folder, then $ROBOREEL_INIT_STATES_DIR, then the repository's assets/data/init_states/."""
    location = init_states_dir or os.environ.get(INIT_STATES_DIR_ENV_VAR) or DEFAULT_INIT_STATES_DIR
    path = Path(location).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Init states folder does not exist: {path}")
    return path


def load_episodes(
    env_id: str,
    init_states_dir: str | Path | None = None,
    distract: bool = False,
    episode_ids: list[int] | None = None,
) -> list[Episode]:
    """Load a task's init states. An episode's id is its position in the file, from 0."""
    path = (
        resolve_init_states_dir(init_states_dir)
        / track_name(distract)
        / f"{TASK_NAMES[env_id]}.pkl"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Init state file does not exist: {path}")
    with open(path, "rb") as f:
        entries = pickle.load(f)

    if episode_ids is None:
        episode_ids = list(range(len(entries)))
    missing = [i for i in episode_ids if not 0 <= i < len(entries)]
    if missing:
        raise ValueError(f"No episodes {missing} in {path}, which has {len(entries)}")

    episodes = []
    for episode_id in episode_ids:
        entry = entries[episode_id]
        num_slots = sum(name.startswith("distractor") for name in entry["state"]["actors"])
        if bool(num_slots) != distract:
            raise ValueError(
                f"Episode {episode_id} in {path} {'has' if num_slots else 'has no'} distractors, "
                f"so it doesn't belong in {track_name(distract)}/."
            )
        # Some files list a different number of names than the state has distractors. The state
        # wins: extra names are dropped, and a missing one repeats the last name.
        names = list(entry.get("distractor_names", []))[:num_slots]
        names = tuple(names + names[-1:] * (num_slots - len(names)))
        episodes.append(Episode(env_id, episode_id, entry["seed"], entry["state"], names))
    return episodes


def human_demo_for_episode(
    env_id: str,
    episode_id: int,
    human_distract: bool = False,
    human_demos_dir: str | Path | None = None,
) -> tuple[int, dict[str, Path]]:
    """Return the human demo id paired with this scene, and its video for each camera."""
    location = (
        human_demos_dir or os.environ.get(HUMAN_DEMOS_DIR_ENV_VAR) or DEFAULT_HUMAN_DEMOS_DIR
    )
    demo_id = HUMAN_DEMO_IDS[episode_id % EPISODES_PER_RUN]
    demo_dir = (
        Path(location).expanduser()
        / "eval"
        / human_track_name(human_distract)
        / TASK_NAMES[env_id]
        / f"{demo_id:04d}"
    )
    videos = {
        camera: demo_dir / f"{camera}.mp4"
        for camera in HUMAN_DEMO_CAMERAS
        if (demo_dir / f"{camera}.mp4").is_file()
    }
    if not videos:
        raise FileNotFoundError(
            f"No human demo videos in {demo_dir}. Download them with scripts/download_human_demos.py."
        )
    return demo_id, videos


def make_env(
    env_id: str,
    obs_mode: str = "rgb",
    control_mode: str = "pd_joint_pos",
    max_steps: int | None = None,
    distract: bool = False,
) -> gym.Env:
    return gym.make(
        env_id,
        obs_mode=obs_mode,
        num_envs=1,
        render_mode="rgb_array",
        robot_uids=ROBOT_UID,
        control_mode=control_mode,
        max_episode_steps=max_steps or MAX_EPISODE_STEPS,
        distract=distract,
    )


def reset_to_episode(env: gym.Env, episode: Episode) -> Any:
    """Put a brand-new env into the episode's saved scene and return the first obs."""
    if episode.distractor_names:
        # The env picks distractors at random, so load the ones this scene was saved with.
        options = {"reconfigure": True, "distractor_names": list(episode.distractor_names)}
        env.reset(seed=episode.seed, options=options)
    else:
        env.reset(seed=episode.seed)
    _check_robot_base(env, episode)
    env.unwrapped.set_state_dict(episode.state)
    return to_numpy(env.unwrapped.get_obs())


def _check_robot_base(env: gym.Env, episode: Episode) -> None:
    """Refuse init states recorded on a different benchmark version's setup."""
    saved = episode.state["articulations"][ROBOT_UID][0, :3].cpu().numpy()
    expected = env.unwrapped.agent.robot.pose.p[0].cpu().numpy()
    if not np.allclose(saved, expected, atol=1e-3):
        raise ValueError(
            f"{episode.env_id} episode {episode.episode_id}: init state has the robot base at "
            f"{saved.round(3).tolist()}, but benchmark {BENCHMARK_VERSION} has it at "
            f"{expected.round(3).tolist()}. This init state is from a different version."
        )


def to_numpy(value: Any) -> Any:
    """Convert a batched ManiSkill observation to numpy, dropping the batch dimension."""
    if isinstance(value, dict):
        return {key: to_numpy(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value[0].cpu().numpy()
    return value


def run_episode(
    policy: Policy,
    episode: Episode,
    obs_mode: str = "rgb",
    control_mode: str = "pd_joint_pos",
    max_steps: int | None = None,
    video_path: Path | None = None,
    human_distract: bool = False,
    human_demos_dir: str | Path | None = None,
) -> tuple[dict, np.ndarray]:
    max_steps = max_steps or MAX_EPISODE_STEPS
    human_demo_id, human_demo = human_demo_for_episode(
        episode.env_id, episode.episode_id, human_distract, human_demos_dir
    )
    # A fresh env per episode, so nothing leaks between episodes.
    env = make_env(
        episode.env_id, obs_mode, control_mode, max_steps, distract=bool(episode.distractor_names)
    )
    frames = []
    actions = []
    success = False
    try:
        obs = reset_to_episode(env, episode)
        policy.reset(episode.episode_id, env, human_demo)
        expected_shape = env.action_space.shape
        for _ in range(max_steps):
            action = np.asarray(policy.act(obs), dtype=np.float32)
            if action.shape != expected_shape:
                raise ValueError(
                    f"Policy returned action shape {action.shape}, expected {expected_shape} "
                    f"for control mode {control_mode}."
                )
            actions.append(action)
            raw_obs, _, _, _, info = env.step(action)
            obs = to_numpy(raw_obs)
            success = bool(info["success"][0])
            if video_path is not None:
                frames.append(to_numpy(env.render()))
            if success:
                break
    finally:
        env.close()

    if video_path is not None:
        import imageio

        imageio.mimsave(video_path, frames, fps=30)

    result = {
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "human_demo_id": human_demo_id,
        "success": success,
        "steps": len(actions),
    }
    return result, np.stack(actions)


def evaluate(
    policy: Policy,
    env_id: str,
    init_states_dir: str | Path | None = None,
    distract: bool = False,
    human_distract: bool = False,
    human_demos_dir: str | Path | None = None,
    itr: int = 1,
    episode_ids: list[int] | None = None,
    obs_mode: str = "rgb",
    control_mode: str = "pd_joint_pos",
    max_steps: int | None = None,
    output_dir: str | Path | None = None,
    save_video: bool = False,
) -> dict:
    """Run the policy and return the results.

    Runs `itr` separate runs of EPISODES_PER_RUN episodes: run 1 uses the first 20 init states,
    run 2 the next 20, and so on. With episode_ids, runs just those episodes as one run instead.
    Each run shows the policy the same 20 human demos, on different scenes.

    An episode succeeds as soon as the task check passes at any step, and stops there.
    """
    max_steps = max_steps or MAX_EPISODE_STEPS
    episodes = load_episodes(env_id, init_states_dir, distract, episode_ids)
    if episode_ids is not None:
        runs = [episodes]
    else:
        if itr < 1 or itr * EPISODES_PER_RUN > len(episodes):
            raise ValueError(
                f"--itr {itr} needs {itr * EPISODES_PER_RUN} episodes, but {env_id} has "
                f"{len(episodes)}. Use a value from 1 to {len(episodes) // EPISODES_PER_RUN}."
            )
        runs = [
            episodes[i * EPISODES_PER_RUN : (i + 1) * EPISODES_PER_RUN] for i in range(itr)
        ]

    # Fail now rather than partway through a long evaluation.
    for episode in episodes:
        human_demo_for_episode(env_id, episode.episode_id, human_distract, human_demos_dir)

    results = {
        "benchmark_version": BENCHMARK_VERSION,
        "env_id": env_id,
        "distract": distract,
        "human_distract": human_distract,
        "obs_mode": obs_mode,
        "control_mode": control_mode,
        "max_steps": max_steps,
        "num_runs": len(runs),
        "num_episodes": sum(len(run) for run in runs),
        "success_rate": None,
        "success_rate_std": None,
        "runs": [],
    }

    env_output_dir = (
        Path(output_dir) / results_track_name(distract, human_distract) / env_id
        if output_dir
        else None
    )
    for run_number, run in enumerate(runs, start=1):
        run_results = {"run": run_number, "success_rate": None, "episodes": []}
        results["runs"].append(run_results)
        for episode in run:
            episode_dir = None
            if env_output_dir:
                episode_dir = env_output_dir / f"run_{run_number}" / f"{episode.episode_id:04d}"
                episode_dir.mkdir(parents=True, exist_ok=True)

            video_path = episode_dir / "video.mp4" if episode_dir and save_video else None
            result, actions = run_episode(
                policy,
                episode,
                obs_mode,
                control_mode,
                max_steps,
                video_path,
                human_distract,
                human_demos_dir,
            )
            run_results["episodes"].append(result)
            successes = [r["success"] for r in run_results["episodes"]]
            run_results["success_rate"] = float(np.mean(successes))
            run_rates = [r["success_rate"] for r in results["runs"]]
            results["success_rate"] = float(np.mean(run_rates))
            results["success_rate_std"] = float(np.std(run_rates))

            # Written after every episode, so a crash keeps the finished ones.
            if env_output_dir:
                np.save(episode_dir / "actions.npy", actions)
                with open(env_output_dir / "results.json", "w") as f:
                    json.dump(results, f, indent=2)

    return results
