# Aloha/Piper 20D train-time RTC serving

`openpi_zmq_server.py` supports both Flexiv 10D and Aloha/Piper 20D EEF policies. RTC dimensionality is checked against the loaded input transform, not inferred from the request. Ordinary prediction remains unchanged.

The portable `aloha_eef_rtc` configuration uses Pi05, model action dimension 32, horizon 50, and maximum trained prefix 8. `LeRobotPiperDataConfig` maps three cameras and uses delta mask `(9,-1,9,-1)`: both xyz/rot6d blocks are relative to current state, both grippers remain absolute. Outputs remove padding and return `[50,20]`.

Before starting, set the template's `AssetsConfig.asset_id` to the directory name under your checkpoint's `assets/` (the template uses `aloha_eef`). For an existing training config, use its name instead; it must use the same Piper transforms, normalization assets and RTC training settings. This portable serving template does not reproduce dataset-specific DAgger training recipes.

```bash
.venv/bin/python openpi_zmq_server.py --addr 'tcp://*:5555' \
  --config aloha_eef_rtc --checkpoint /path/to/checkpoint
```

Request `cmd=predict_rtc` with physical 20D `state`, image keys `observation/image`, `observation/left_wrist_image`, `observation/right_wrist_image`, and `rtc={prefix_length:d,prefix_actions:[d,20]}`. Use 0 <= d <= 8; the initial empty prefix is `[]`. State/action order is `[left xyz, left rot6d, left gripper, right xyz, right rot6d, right gripper]`, with rot6d storing the first two matrix columns.

The existing Policy pipeline transforms the physical prefix through current-state deltas, normalization and model padding. Every denoising step clamps the selected prefix at t=0. Zero-prefix requests use the ordinary sampler. The response confirms `applied:true`, `mode:train_time_prefix`, the requested prefix length, and maximum prefix length 8. Returned physical prefix values exactly match the request.

CPU regression tests cover both layouts, zero-prefix equivalence, 4/8-step requests, wrong-layout rejection and dual-arm normalization:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q scripts/test_rtc_inference.py
```

Client: [InferSystem Aloha RTC](https://github.com/Celoond/InferSystem/blob/main/docs/aloha_rtc.md).
