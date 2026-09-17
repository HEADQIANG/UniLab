# Mixed PPO task profiles

The `mixed` PPO owners use the corresponding MuJoCo task as the common policy
contract. Go2 flat provides MuJoCo, Motrix, and Drake worker/evaluation leaves;
G1 flat provides MuJoCo, Motrix, MuJoCo Warp, IsaacGym, IsaacSim, Genesis, and
Newton leaves. A leaf is named `<backend>_mixed.yaml` and does not inherit the
original backend's separately tuned action, observation, reward, or PPO settings.

All mixed profiles disable `env.events.pd_gains` and
`env.events.reset_root_state_uniform`. The remaining reset uses the scene's
default keyframe. Go2 also disables `reward.contact`; G1 disables
`reward.feet_air_time`. These tasks have no separate `env.domain_rand` block.
Observation noise, command sampling, action scaling, control timing, termination
rules, and network/normalization settings remain those of the MuJoCo owner.
Original single-backend owner configurations are unchanged.

The mixed training owner declares `training.mix_task` for CLI task selection,
selects the mixed runtime resolver for the parent learner, and disables automatic
playback. Each evaluation leaf restores its physical backend, clears the mixed
runtime resolver to `null`, and selects that backend's render mode. Workers do not
instantiate the mixed learner resolver themselves. Use the CLI's
`--profile mixed` with the chosen physical backend when evaluating a mixed policy.
Backend device, solver, and capacity settings remain explicit in these leaves;
the presence of a leaf does not establish runtime installation or convergence.
`--profile mixed` rejects unsupported task/backend pairs and missing owner files;
it never substitutes another backend's owner configuration.

`--mix-config` accepts Unicode paths and paths containing spaces, quotes, or
backslashes. Quote the path as one shell argument; the CLI preserves the path
when passing it to Hydra. Relative paths are resolved against the launch
directory, including when the checkout itself has a non-ASCII path:

```bash
uv run train --algo ppo --task go2_joystick_flat \
  --sim-mix mujoco=0.7,motrix=0.3 \
  --mix-config 'docs/examples/mixed_go2.yaml'
```

For an explicit checkpoint path, use `algo.load_run`, not `--load-run` (which
accepts a run name). For example, a short headless MuJoCo evaluation of a policy
trained with `[32, 32]` actor and critic layers is:

```bash
MUJOCO_GL=egl uv run eval --algo ppo --task go2_joystick_flat \
  --sim mujoco --profile mixed --render-mode record \
  algo.load_run=/absolute/path/to/model_1.pt \
  training.device=cpu training.play_env_num=1 training.play_steps=2 \
  'algo.policy.actor_hidden_dims=[32,32]' \
  'algo.policy.critic_hidden_dims=[32,32]'
```

Network shape overrides must match the training run. Omit the two overrides for
default-size policies. Offline video is written alongside the checkpoint;
MuJoCo's EGL renderer and the repository's video/export dependencies must be
installed. Replace `--sim mujoco` with `--sim motrix` for the other Go2 CPU
backend. Retain `--profile mixed` in either case.

After installing the project's test dependencies, run from the repository root:

```bash
uv run pytest tests/config/test_mixed_profiles.py tests/scripts/test_mixed_cli.py -q
```

The tests compose all ten legal worker/evaluation profiles, compare the complete
ordered environment contract and rewards, check runtime-only differences, and
verify that original single-backend tuning remains available.
