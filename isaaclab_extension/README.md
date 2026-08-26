# Isaac Lab handoff

The repository is ready to validate the learned controller in Isaac Lab, but
the Isaac runtime is intentionally not vendored. Isaac Lab should be run on a
Linux or Windows machine with an NVIDIA RTX GPU.

The shared contract is `ewm/isaac_bridge.py` and
`configs/single_arm_isaac_contract.json`:

- observation: the 53 SingleArm `observation.state` values in the exact listed order;
- action: 7 values `(ee_pos_delta_xyz, ee_rot_vec_delta_xyz, gripper)`;
- task success: box 0 is within 6 cm horizontally of box 1 and its vertical gap is 3–9 cm;
- horizon: 50 control steps.

## Preflight

From the project root:

```bash
python scripts/preflight_isaac.py
```

On an Isaac machine, install Isaac Sim and Isaac Lab using their official
version-matched instructions, then create a `DirectRLEnv` that:

1. loads a Franka Panda and two cubes;
2. applies the 7-D end-effector delta action through the Panda task's IK/controller;
3. emits the 53-dimensional state in the contract order;
4. uses the same reset seed and stacking reward/success rule;
5. exports rollout videos and episode metrics.

The offline learned-simulator benchmark remains the pre-Isaac regression test:

```bash
python scripts/evaluate_model_env.py artifacts/single_arm_100ep.pkl data/single_arm/panda-stack-wide --controller cem --episodes 10 --max-steps 50
```

The first Isaac experiment should compare the saved CEM action sequences and a
random controller in the real simulator. Do not compare absolute success
numbers across the learned simulator and Isaac until reset distributions,
action scales, and cube geometry are aligned.
