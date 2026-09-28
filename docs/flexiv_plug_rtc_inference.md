# Flexiv train-time RTC 服务端

支持 InferSystem 的普通 `predict` 和 `predict_rtc` 请求。JAX pi0.5 支持 clean-prefix 采样；不支持 PyTorch RTC 或 inference-time guidance。

```bash
uv pip install pyzmq msgpack
bash scripts/serve_flexiv_plug_rtc.sh /path/to/checkpoint
```

Checkpoint 需要 `params` 和 `assets/insert-plug-bc/norm_stats.json`，不需要 `train_state`。
`flexiv_plug_rtc` 是与该 checkpoint 对应的服务配置，不提供训练数据或权重。
state/action 都是 10 维 xyz + rot6d + gripper；对应数据的夹爪单位为 10×米，客户端应分别正确换算输入和输出。

## 请求协议

ZMQ REQ/REP + msgpack；图像可用 JPEG bytes，颜色处理与原服务相同。

```python
request = {
    'cmd': 'predict_rtc',
    'state': current_eef_state.tolist(),  # [10]
    'observation/image': main_jpeg,
    'observation/wrist_image': wrist_jpeg,
    # 可选 observation/secondary_image；缺省为黑图且 image mask=False
    'rtc': {
        'prefix_length': d,              # Python int, 0 <= d <= 8
        'prefix_actions': committed_prefix.tolist(),  # [d, 10]
    },
}
```

首个请求没有旧动作：`rtc={'prefix_length': 0}`，此时生成普通完整 chunk。
后续前缀必须使用机器人执行空间的**绝对 EEF 动作**，布局 `[x,y,z,r1,...,r6,gripper]`，与服务返回的 `actions` 相同。不要发送已归一化动作或相对上次 state 的 delta。

服务会相对**本次观测的 state** 对前 9 维做差，夹爪保持绝对值，使用 checkpoint 自带的 `assets/insert-plug-bc/norm_stats.json` 做 quantile normalization，最后补到 32 维。

响应示例：

```python
{
    'status': 'ok',
    'actions': ...,  # [50, 10]，完整 chunk，包含固定前缀
    'rtc': {
        'applied': True,
        'mode': 'train_time_prefix',
        'prefix_length': d,
        'max_prefix_length': 8,
    },
    'infer_time_ms': ...,
    'model_infer_time_ms': ...,
}
```

非法长度、维度、非有限数值、缺失前缀会返回 `status='error'`。普通 `predict` 如果夹带 `rtc` 字段也会报错，避免静默忽略。此前 inference-time guidance 协议的 `leftover` / `guidance_weight` / `prefix_schedule` 等字段不兼容本接口。

## 客户端队列对齐

1. 以本次观测对应的执行步为新 chunk 的第 0 步，同步记录已经实际发布/执行的步数。
2. 从旧队列取出接下来仍会执行的 `d` 步作为前缀。请求发送后，这些动作继续按原队列执行。
3. 响应到达时，根据从观测时刻起**实际已消耗的步数** `k` 丢弃新 chunk 的前 `k` 步，再按同一时间轴接续。若恰好 `k=d`，接上 `actions[d:]`。不要把固定前缀再执行一遍，也不要一律只丢弃 `d` 步而忽略真实延迟。
4. 如果实际延迟超过承诺前缀长度，超出的已执行动作没有作为条件提供，连续性不再有保证；客户端需要重新对齐/重请求或采用自己的等待策略。不能静默把超过 8 步的需求裁到 8，也不能假设 `推理秒数 × FPS` 就是实际执行步数。
5. Reset 不保存/恢复旧前缀；新 episode 的客户端必须清空旧队列并从 `d=0` 开始。服务不在请求之间隐式缓存动作，避免跨 episode 混入前缀。

这里是与训练对应的 clean-prefix 条件采样：前缀动作每一步保持不变、时间为 0；后缀从噪声按 Euler 法去噪，默认 10 步。模型结构和参数形状不变，Gemma adaRMS 支持逐 token 时间。`prefix_length` 作为动态 JAX 标量，1–8 步使用相同形状；零前缀复用原始采样路径。首次调用涉及 JIT 编译，正式异步执行前应先做离线 warmup。

## 验证

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q scripts/test_rtc_inference.py src/openpi/models/pi0_test.py
# 可选：需要用户自己的 checkpoint 和空闲 GPU；合成观测，不连接机器人。
.venv/bin/python scripts/smoke_rtc_checkpoint.py --checkpoint /path/to/checkpoint
```

零前缀复用普通采样路径；非零前缀经当前 state 的 delta/归一化转换后，每一步采样均保持固定。
InferSystem 2.0 前缀模式在请求期间继续消费旧队列，并按实际发布步数裁剪、融合新结果。
全段融合、请求频率和前缀是不同因素，服务端不会替客户端管理执行队列。
