# Spider PPO in PyBullet

Train a 12-DoF spider robot to walk forward in PyBullet using PPO from
Stable-Baselines3. The environment exposes joint position control, dense
observations, and a forward-progress reward with survival and energy terms.

## Demo (GIF Placeholder)

![Spider walking demo](demo.gif)

Replace `docs/demo.gif` with your recording.

## Overview

- **Policy**: PPO (`MlpPolicy`) from Stable-Baselines3.
- **Environment**: custom Gymnasium env backed by PyBullet.
- **Vectorized training**: 4 parallel envs with `SubprocVecEnv` + `VecNormalize`.
- **Artifacts**:
	- `ppo_spider`: trained policy.
	- `vec_normalize.pkl`: observation/reward normalization stats.
	- `best_models/`: best checkpoints from eval callback.

## Environment Details

- **Action space**: 12-D continuous box in $[-0.785, 0.785]$.
	- Each action is added to the current joint angle (scaled by 0.1) and then
		clipped to $[-0.785, 0.785]$.
- **Observation space**: 37 floats
	- Base position (3), base orientation quaternion (4), linear velocity (3),
		angular velocity (3), and 12 joint positions/velocities (24).
- **Reward**:
	- Forward progress $50 \Delta x$
	- Alive bonus $+0.1$ if $\Delta x > 0.005$
	- Energy penalty $-0.01 \sum |a_i|$
	- Fall penalty $-5$ if base height $z < 0.03$
- **Episode end**:
	- Truncated at 1000 steps
	- Terminated if base height drops below 0.03

## Project Structure

- `env.py`: Gymnasium environment and env factory.
- `train.py`: PPO training + evaluation + test rollouts.
- `urdf/`: spider model and meshes.
- `best_models/`: best checkpoints saved during training.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Train

`train.py` currently runs `test()` by default. To train, switch the entry point:

```python
if __name__ == "__main__":
		train()
		# test()
```

Then run:

```bash
python train.py
```

## Test / Rollout

By default, running `python train.py` executes `test()` and expects a saved
`ppo_spider` model (and optionally `vec_normalize.pkl`). This opens a PyBullet
GUI window and runs the policy in a loop.

## Notes

- The environment uses `p.GUI`, so training with multiple subprocesses will
	open multiple windows. For headless training, switch to `p.DIRECT` in `env.py`.
- `VecNormalize` stats are required for consistent evaluation; keep
	`vec_normalize.pkl` alongside the model.

## Theory (Fill In)

- PPO clipped objective and why it stabilizes updates.
- Generalized Advantage Estimation (GAE) and bias-variance tradeoff.
- Why action scaling/clipping is needed for position control.
- The link between reward shaping and gait emergence.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).
If you distribute a modified version, you must keep it under GPL-3.0 and
provide the source code. See <https://www.gnu.org/licenses/gpl-3.0.html>.
