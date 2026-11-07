# RoboReel

[![arXiv](https://img.shields.io/badge/arXiv-2609.08209-b31b1b.svg)](https://arxiv.org/abs/2609.08209)
[![Website](https://img.shields.io/badge/Website-roboreel.github.io-blue.svg)](https://roboreel.github.io/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/LogosRoboticsLab/roboreel)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

RoboReel is a simulated robot manipulation benchmark built on
[ManiSkill](https://github.com/haosulab/ManiSkill). It has 10 tabletop tasks for a Franka Panda
arm. Your policy watches a video of a human doing a task, then has to do that task in simulation.
Every method uses the same videos and the same fixed starting scenes, so results are directly
comparable.

Bring any algorithm: the benchmark restores the scene, gives your policy the human video and
observations, applies its actions and records the results.

## Contents

- [Tasks](#tasks)
- [Installation](#installation)
- [Data](#data)
- [Quick start](#quick-start)
- [Evaluate your policy](#evaluate-your-policy)
- [Observations and actions](#observations-and-actions)
- [Distractors, videos and runs](#distractors-videos-and-runs)
- [Evaluation protocol](#evaluation-protocol)
- [Benchmark versions](#benchmark-versions)
- [Citation](#citation)
- [License](#license)

## Tasks

| Task ID | Goal |
|---|---|
| `PickCube-v2` | Lift the cube and hold the robot still |
| `PushCube-v2` | Push the cube to the goal |
| `CloseDrawer-v1` | Close the open drawer |
| `Toaster-v1` | Press the toaster's slider |
| `BowlOnPlate-v1` | Put the bowl on the plate |
| `OpenLid-v1` | Take the lid off and put it on the table |
| `PutBasket-v1` | Put both objects in the basket |
| `EmptyBasket-v1` | Take both objects out of the basket |
| `StackCups-v1` | Stack the cups: dark blue, then red, then light blue on top |
| `ObjectInDrawer-v1` | Put the object in the drawer and close it |

Each episode can run for up to 1000 steps.

## Installation

You need Linux, an NVIDIA GPU, [Docker](https://docs.docker.com/engine/install/) and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The Docker image takes about 45 GB of disk space.

### 1. Build the image

```bash
git clone https://github.com/gupta-anmol99/roboreel.git
cd roboreel/docker
docker compose build
```

### 2. Start the container

```bash
cd docker
docker compose run --rm roboreel
```

Your repository folder appears inside the container as `/workspace`, and the following commands
run there. Files the container creates on your machine, such as results, are owned by `root`.

### 3. Download the PartNet-Mobility models

`CloseDrawer-v1`, `ObjectInDrawer-v1` and `Toaster-v1` each use one model from
[PartNet-Mobility](https://sapien.ucsd.edu/downloads). The other tasks don't need this step.

| Used by | Model ID |
|---|---|
| `CloseDrawer-v1`, `ObjectInDrawer-v1` | `44962` |
| `Toaster-v1` | `103498` |

Register on the SAPIEN website and get your access token. Then, inside the container, run this with
your token:

```bash
python -c "
import sapien.asset
for model_id in (44962, 103498):
    sapien.asset.download_partnet_mobility(model_id, token='<YOUR_TOKEN>', directory='assets/data/dataset')
"
```

You should end up with:

```
assets/data/dataset/
├── 44962/
│   └── mobility.urdf
└── 103498/
    └── mobility.urdf
```

## Data

| Data | Location | Use |
|---|---|---|
| Test scenes | `assets/data/init_states/` in this repository | Starting state of each evaluation scene |
| Human videos, episodes `0080`-`0099` | [Hugging Face](https://huggingface.co/datasets/LogosRoboticsLab/roboreel) | The videos your policy watches during evaluation |
| Training data | [Hugging Face](https://huggingface.co/datasets/LogosRoboticsLab/roboreel) | Not used by the evaluation scripts |

Each task has 20 evaluation videos with distractors and 20 without, and each video has four cameras:
`egocentric`, `front`, `left` and `right`.

### Download the human videos

Inside the container, run:

```bash
python scripts/download_human_demos.py
```

This downloads all the evaluation videos, about 11 GB, into `human_demos/eval/`. To download less,
choose what you need:

```bash
python scripts/download_human_demos.py --distraction no --tasks open_lid --cameras front
```

## Quick start

Download the videos for one task, then run a random policy on one episode of it:

```bash
python scripts/download_human_demos.py --distraction no --tasks empty_basket --cameras front
```

```bash
python scripts/evaluate.py \
    --env EmptyBasket-v1 \
    --policy examples.random_policy:RandomPolicy \
    --episodes 0 \
    --save-video
```

To use a different folder, pass `--init-states-dir` or set `ROBOREEL_INIT_STATES_DIR`. Inside the
container, the folder must be somewhere under `/workspace`, unless you mount it yourself.

Results are saved like this:

```
results/human_no_distraction__env_without_distraction/EmptyBasket-v1/
├── results.json        # success rates and per-episode results
└── run_1/
    └── 0000/
        ├── actions.npy # every action the policy took
        └── video.mp4   # with --save-video
```

```json
{
  "benchmark_version": "v2",
  "env_id": "EmptyBasket-v1",
  "distract": false,
  "human_distract": false,
  "obs_mode": "rgb",
  "control_mode": "pd_joint_pos",
  "max_steps": 1000,
  "num_runs": 1,
  "num_episodes": 1,
  "success_rate": 0.0,
  "success_rate_std": 0.0,
  "runs": [
    {
      "run": 1,
      "success_rate": 0.0,
      "episodes": [
        {
          "episode_id": 0,
          "seed": 1,
          "human_demo_id": 80,
          "success": false,
          "steps": 1000
        }
      ]
    }
  ]
}
```

## Evaluate your policy

### Option 1: write a `Policy` class

This is the simplest option and needs two methods. See
[`examples/random_policy.py`](examples/random_policy.py).

```python
# my_policy.py
from benchmark import Policy


class MyPolicy(Policy):
    def __init__(self, checkpoint):
        self.model = load_my_model(checkpoint)

    def reset(self, episode_id, env, human_demo):
        # Called before each episode. human_demo maps each camera to its video file.
        self.model.reset()
        self.model.watch(human_demo["front"])

    def act(self, obs):
        # One observation in, one action out.
        return self.model.predict(obs["sensor_data"]["base_camera"]["rgb"])
```

Run it with:

```bash
python scripts/evaluate.py \
    --env EmptyBasket-v1 \
    --policy my_policy:MyPolicy \
    --policy-kwargs '{"checkpoint": "model.pt"}' \
```

Useful options:

| Option | Meaning | Default |
|---|---|---|
| `--distract` | Use the scenes with distractors | Off |
| `--human-distract` | Use the human videos with distractors | Off |
| `--human-demos-dir` | Folder holding the videos' `eval/` folder. You can also set `ROBOREEL_HUMAN_DEMOS_DIR` | `human_demos` |
| `--itr` | Number of runs of 20 episodes | `1` |
| `--episodes 0 1 2` | Run only these episodes, as one run. Don't use this for reported results | Off |
| `--obs-mode` | `state`, `rgb` or `rgbd` | `rgb` |
| `--control-mode` | `pd_joint_pos` or `pd_ee_delta_pose` | `pd_joint_pos` |
| `--output-dir` | Where to save results | `results` |
| `--save-video` | Save a video of each episode | Off |
| `--max-steps` | Override the step limit. Don't use this for reported results | `1000` |

### Run every task

Use `--env all` to run all 10 tasks, or list the task IDs you want:

```bash
python scripts/evaluate.py \
    --env all \
    --policy my_policy:MyPolicy \
```

This also saves `summary.json` in the results folder, with each task's success rate and the mean
across the tasks you ran. The same policy object is used for every task, and `env.spec.id` in
`reset()` tells you the current task. If you train one model per task, load the right one there or
run each task separately.

### Option 2: write your own loop

Use this if your code needs to control the loop itself, for example to batch requests to a model
server. Run it with `PYTHONPATH=src`.

```python
import numpy as np
from benchmark import (
    EPISODES_PER_RUN,
    MAX_EPISODE_STEPS,
    human_demo_for_episode,
    load_episodes,
    make_env,
    reset_to_episode,
    to_numpy,
)

env_id = "EmptyBasket-v1"
distract = False
successes = []

for episode in load_episodes(env_id, distract=distract)[:EPISODES_PER_RUN]:
    # Use a new env for each episode, so nothing carries over between episodes.
    env = make_env(env_id, obs_mode="rgb", control_mode="pd_joint_pos", distract=distract)
    obs = reset_to_episode(env, episode)
    _, human_demo = human_demo_for_episode(env_id, episode.episode_id)  # give this to your policy

    for _ in range(MAX_EPISODE_STEPS):
        action = env.action_space.sample()  # replace with your policy
        obs, reward, terminated, truncated, info = env.step(action)
        obs = to_numpy(obs)
        # An episode succeeds as soon as the check passes (see "Evaluation protocol").
        if info["success"][0]:
            break

    successes.append(bool(info["success"][0]))
    env.close()

print("Success rate:", np.mean(successes))
```

## Observations and actions

### Observations

`obs` is a nested dictionary of NumPy arrays with no batch dimension. With `--obs-mode rgb`:

| Key | Contents |
|---|---|
| `obs["sensor_data"][camera]["rgb"]` | Image; see the camera table for its size |
| `obs["sensor_param"][camera]` | Camera intrinsics and extrinsics |
| `obs["agent"]["qpos"]`, `obs["agent"]["qvel"]` | Joint positions and velocities |
| `obs["extra"]` | Task-specific values, such as `tcp_pose`, the gripper's pose |

With `--obs-mode state`, `obs` is instead one flat array.

The cameras are:

| Camera | Mount | Position (x, y, z) | Image size (height, width) | Field of view |
|---|---|---|---|---:|
| `base_camera` | Fixed | 0.618, -0.005, 0.729 | 288, 512 | 43° |
| `left_camera` | Fixed | -0.706, 0.670, 0.732 | 288, 512 | 43° |
| `right_camera` | Fixed | -0.769, -0.803, 0.686 | 288, 512 | 43° |
| `overhead_camera` | Fixed | 0, 0, 0.6 | 512, 512 | 90° |
| `hand_camera` | On the gripper | Moves with the gripper | 512, 512 | 90° |

The robot's base is at x = -0.95, y = 0, z = -0.20 in every task: beside the table, 20 cm below its surface.

### Actions

`act()` returns one NumPy array, and its shape depends on the control mode:

| Control mode | Shape | Contents |
|---|---|---|
| `pd_joint_pos` | `(8,)` | 7 target joint positions in radians, then the gripper |
| `pd_ee_delta_pose` | `(7,)` | Change in gripper position (x, y, z) and rotation (axis-angle), scaled to `[-1, 1]`, then the gripper |

The gripper value runs from `-1` (closed) to `1` (open).

In `pd_ee_delta_pose`, each step's command is relative to the gripper's current pose, along the
robot base's axes. A value of `1` asks for 0.1 m or 0.1 rad in that step. Large commands may not be
reached in a single step: at `0.5` (5 cm), the gripper moved about 2 cm per step in our test.

Choose the mode with `--control-mode`. It changes how a policy acts but not the task. Report which
one you used; it's saved in `results.json`. Each mode writes to the same output folder by default,
so use a different `--output-dir` for each mode to keep both sets of results.

## Distractors, videos and runs

The scene and the human video can each have distractor objects or not, which gives four evaluation
settings with the same tasks, success checks and scoring. Report them separately:

| Human video | Scene | Options |
|---|---|---|
| No distractors | No distractors | None |
| No distractors | Distractors | `--distract` |
| Distractors | No distractors | `--human-distract` |
| Distractors | Distractors | `--human-distract --distract` |

Scenes with distractors have 1 or 2 extra objects on the table. These can't be moved, but the robot
can collide with them.

Scenes are in `assets/data/init_states/without_distraction/` and `assets/data/init_states/with_distraction/`. Each folder has one file per task. `Toaster-v1` is in `press_toaster.pkl`, `BowlOnPlate-v1` in
`bowl_in_plate.pkl`, `PutBasket-v1` in `toys_in_basket.pkl` and `ObjectInDrawer-v1` in
`toys_in_drawer.pkl`. The other files are named after their task, such as `open_lid.pkl`.

Distractors can't be moved, but the robot can collide with them.

`--itr` sets how many runs of 20 episodes to do, in file order. Every run uses the same 20 human
videos, `0080`-`0099`, on different scenes: a run's first scene gets video `0080`, its second gets
`0081`, and so on.

| `--itr` | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| `1` | Scenes 0-19 | | |
| `2` | Scenes 0-19 | Scenes 20-39 | |
| `3` | Scenes 0-19 | Scenes 20-39 | Scenes 40-59 |

The largest value is `5`. A task's `success_rate` is the mean of its runs, and `success_rate_std` is
their standard deviation.

To evaluate one setting on every task:

```bash
python scripts/evaluate.py \
    --env all \
    --policy my_policy:MyPolicy \
    --human-distract \
    --distract \
    --itr 3
```

## Evaluation protocol

To report comparable results:

1. Don't train or tune on the evaluation videos or test scenes.
2. Use the unchanged test scenes and videos, and run each task with `--itr 3`, for all four settings.
3. Stop each episode as soon as the task's success check passes, or at the 1000-step limit.
4. Count an episode as a success if the check passes at **any step**. Each episode is a success or a
   failure, with no partial credit.
5. Report `success_rate` and `success_rate_std` per task and the mean across tasks, for each setting,
   along with the benchmark version, observation mode and control mode.

In a run of 20 episodes, one episode is worth 5 percentage points on a task, so treat small
differences with care.

`steps` in `results.json` is how many steps the episode ran. On a success, that's when the check
passed. It doesn't count toward the score.

### Known limits of the v1 success checks

v1 keeps its success checks unchanged so that results stay comparable. Keep these limits in mind
when you read scores:

- **Any task:** the check only has to pass once, so a brief pass counts even if the result wouldn't
  have lasted.
- **`EmptyBasket-v1`:** objects are measured from the basket's current position, so moving the
  basket away from an object, or lifting an object just above it, can count as taking it out.
- **`Toaster-v1`:** it passes after about 18% of the slider's travel, once the slider has stopped.
- **`BowlOnPlate-v1`, `StackCups-v1` and `ObjectInDrawer-v1`:** the robot doesn't have to let go, so
  an object held in the right place can count.
- **`CloseDrawer-v1` and `ObjectInDrawer-v1`:** the drawer counts as closed once it has moved half of
  its travel.
- **`PickCube-v2`:** the cube only has to be 3 cm above its starting height.

## Benchmark versions

| Version | Table | Cameras | Status |
|---|---|---|---|
| `v1` | Default ManiSkill wood table | Setup used in the published paper | Released |
| `v2` | Black table | Calibrated cameras | In development |

Init states only work with the version they were made for. The benchmark raises an error if an
init state's robot position doesn't match the current version.

## Citation

```bibtex
@misc{gu2026monkeyseemonkeydo,
  title         = {Monkey See, Can Monkey Do? A Benchmark for Evaluating Robot Skill Learning by Observation},
  author        = {Weiwei Gu and Anmol Gupta and Anant Sah and Ryan Varghese and Lalitha Shreya Vanam and Prabhath Adireddi and Peter Karkus and Nakul Gopalan},
  year          = {2026},
  eprint        = {2609.08209},
}
```

## License

The code in this repository is released under the [MIT License](LICENSE). PartNet-Mobility, ManiSkill
and other third-party components have their own terms.
