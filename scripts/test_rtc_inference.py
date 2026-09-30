"""CPU regression tests for train-time RTC."""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import pytest
from openpi import transforms as tr
from openpi.models import gemma, pi0
from openpi.models.model import ModelType
from openpi.policies.policy import Policy
from openpi.policies.flexiv_policy import FlexivInputs, FlexivOutputs
from openpi.policies.piper_policy import PiperInputs, PiperOutputs
from openpi_zmq_server import PIZmqServer


def make_policy(action_dim=10):
    p = object.__new__(Policy)
    p._model = SimpleNamespace(pi05=True, rtc_training_max_delay=8, action_horizon=50)
    p._is_pytorch_model = False
    p._rng = jax.random.key(0)
    p._sample_kwargs = {}
    stats = {k: tr.NormStats(mean=np.zeros(action_dim), std=np.ones(action_dim), q01=-np.ones(action_dim), q99=np.ones(action_dim)*3)
             for k in ('state', 'actions')}
    inputs_type, outputs_type = (FlexivInputs, FlexivOutputs) if action_dim == 10 else (PiperInputs, PiperOutputs)
    delta_mask = tr.make_bool_mask(*((9, -1) * (action_dim // 10)))
    p._input_transform = tr.compose([inputs_type(ModelType.PI05), tr.DeltaActions(delta_mask),
                                     tr.Normalize(stats, use_quantiles=True), tr.PadStatesAndActions(32)])
    p._output_transform = tr.compose([tr.Unnormalize(stats, use_quantiles=True),
                                      tr.AbsoluteActions(delta_mask), outputs_type()])
    calls = []
    def sample(rng, obs, **kwargs):
        calls.append(kwargs)
        return kwargs.get('rtc_prefix', jnp.zeros((1, 50, 32)))
    p._sample_actions = sample
    return p, calls


def observation(action_dim=10):
    if action_dim == 20:
        return {'state': np.arange(20, dtype=np.float32)/10,
                **{key: np.zeros((224,224,3), np.uint8) for key in
                   ('observation/image', 'observation/left_wrist_image', 'observation/right_wrist_image')}}
    return {'state': np.arange(10, dtype=np.float32)/10,
            'observation/image': np.zeros((224,224,3), np.uint8),
            'observation/wrist_image': np.zeros((224,224,3), np.uint8)}


@pytest.mark.parametrize("action_dim", [10, 20])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_prefix_transforms_current_state_and_roundtrip(dtype, action_dim):
    p, calls = make_policy(action_dim)
    obs = observation(action_dim)
    prefix = np.tile(np.arange(action_dim, dtype=dtype) + 0.123456789, (8,1))
    original = prefix.copy()
    out = p.infer(obs, rtc_prefix=prefix, rtc_prefix_length=8)
    delta = prefix-obs['state']; delta[:,9::10] = prefix[:,9::10]
    expected = (delta+1)/(4+1e-6)*2-1
    np.testing.assert_allclose(calls[0]['rtc_prefix'][0,:8,:action_dim], expected, atol=1e-6)
    np.testing.assert_array_equal(calls[0]['rtc_prefix'][0,:8,action_dim:], 0)
    np.testing.assert_array_equal(out['actions'][:8], original)
    np.testing.assert_array_equal(prefix, original)
    assert out['rtc']['applied'] and out['actions'].shape == (50,action_dim)
    obs['state'] += 0.2
    p.infer(obs, rtc_prefix=prefix, rtc_prefix_length=8)
    assert not np.allclose(calls[0]['rtc_prefix'], calls[1]['rtc_prefix'])


@pytest.mark.parametrize('length', [-1, 9, True, 1.5])
def test_invalid_lengths(length):
    p, _ = make_policy()
    with pytest.raises(ValueError):
        p.infer(observation(), rtc_prefix_length=length)


@pytest.mark.parametrize('prefix', [np.zeros((7,10)), np.zeros((8,32)), np.full((8,10), np.nan)])
def test_invalid_prefix(prefix):
    p, _ = make_policy()
    with pytest.raises(ValueError):
        p.infer(observation(), rtc_prefix=prefix, rtc_prefix_length=8)


@pytest.mark.parametrize("action_dim", [10, 20])
def test_wire_protocol_bootstrap_and_fail_closed(action_dim):
    server = PIZmqServer()
    server._policy, _ = make_policy(action_dim)
    msg = {k:v.tolist() for k,v in observation(action_dim).items()}
    def send(**fields):
        return server._handle_request(msgpack.packb({**msg, **fields}))
    bootstrap = send(cmd='predict_rtc', rtc={'prefix_length':0})
    assert bootstrap['rtc']['applied'] and bootstrap['rtc']['prefix_length'] == 0
    assert send(cmd='predict')['status'] == 'ok'
    for fields in [dict(cmd='predict',rtc={'prefix_length':0}), dict(cmd='predict_rtc'),
                   dict(cmd='predict_rtc',rtc={'prefix_length':8}),
                   dict(cmd='predict_rtc',rtc={'prefix_length':0,'guidance_weight':1})]:
        assert send(**fields)['status'] == 'error'
    server._policy._model.rtc_training_max_delay = 0
    assert send(cmd='predict_rtc',rtc={'prefix_length':0})['status'] == 'error'
    assert server._handle_request(msgpack.packb([]))['status'] == 'error'


def test_sampling_clamps_prefix_at_every_step_and_conditions_postfix(monkeypatch):
    monkeypatch.setattr(pi0._model, 'preprocess_observation', lambda _, obs, **kw: obs)
    times = []
    def suffix(obs, actions, t):
        jax.debug.callback(lambda v: times.append(np.array(v)), t)
        velocity = actions + actions.mean(axis=1, keepdims=True) + t[...,None]
        return velocity, jnp.ones((1,5),bool), jnp.array([True,False,False,False,False]), None
    model = SimpleNamespace(pi05=True,rtc_training_max_delay=2,action_horizon=5,action_dim=3,
        embed_prefix=lambda obs:(jnp.zeros((1,1,3)),jnp.ones((1,1),bool),jnp.zeros((1,),bool)),
        embed_suffix=suffix,PaliGemma=SimpleNamespace(llm=lambda xs,**kw:(xs,None)),action_out_proj=lambda x:x)
    obs = SimpleNamespace(state=jnp.zeros((1,3)))
    noise = jnp.ones((1,5,3)); prefix = noise*7
    result = pi0.Pi0.sample_actions(model,jax.random.key(0),obs,num_steps=3,noise=noise,
                                    rtc_prefix=prefix,rtc_prefix_length=jnp.array(2))
    result.block_until_ready()
    np.testing.assert_array_equal(result[:,:2], prefix[:,:2])
    assert len(times)==3
    for t in times:
        np.testing.assert_array_equal(t[:,:2], 0)
        assert np.all(t[:,2:] > 0)
    other = pi0.Pi0.sample_actions(model,jax.random.key(0),obs,num_steps=3,noise=noise,
                                   rtc_prefix=prefix*2,rtc_prefix_length=jnp.array(2))
    assert not np.allclose(result[:,2:], other[:,2:])


def test_token_time_matches_scalar_time():
    scalar = jnp.array([0.5, 0.2])
    embedding = pi0.posemb_sincos(scalar,16,0.004,4.0)
    token_embedding = pi0.posemb_sincos(jnp.tile(scalar[:,None],(1,5)),16,0.004,4.0)
    np.testing.assert_array_equal(token_embedding,jnp.tile(embedding[:,None],(1,5,1)))
    norm = gemma.RMSNorm(); x = jnp.ones((2,5,16))
    variables = norm.init(jax.random.key(0),x,embedding)
    for a,b in zip(norm.apply(variables,x,embedding),norm.apply(variables,x,token_embedding)):
        np.testing.assert_allclose(jnp.broadcast_to(a,b.shape),b)


@pytest.mark.parametrize("action_dim", [10, 20])
@pytest.mark.parametrize("length", [0, 4, 8])
def test_wire_rtc_layout_matches_loaded_policy(action_dim, length):
    server = PIZmqServer()
    server._policy, calls = make_policy(action_dim)
    obs = observation(action_dim)
    prefix = np.tile(obs["state"], (length, 1)).astype(np.float64)
    prefix[:, 9::10] = -0.02
    request = {k: v.tolist() for k, v in obs.items()}
    request.update(cmd="predict_rtc", rtc={"prefix_length": length, "prefix_actions": prefix.tolist()})
    out = server._handle_request(msgpack.packb(request))
    assert out["status"] == "ok"
    assert np.asarray(out["actions"]).shape == (50, action_dim)
    np.testing.assert_array_equal(np.asarray(out["actions"])[:length], prefix)
    assert out["rtc"]["prefix_length"] == length and out["rtc"]["applied"]
    assert ("rtc_prefix" in calls[-1]) == (length > 0)
    if length == 0:
        ordinary = server._handle_request(msgpack.packb({**request, "cmd": "predict", "rtc": {}}))
        assert ordinary["status"] == "error"
        plain = {k: v for k, v in request.items() if k != "rtc"}
        plain["cmd"] = "predict"
        np.testing.assert_array_equal(out["actions"], server._handle_request(msgpack.packb(plain))["actions"])
    request["state"] = [0.] * (20 if action_dim == 10 else 10)
    count = len(calls)
    bad = server._handle_request(msgpack.packb(request))
    assert bad["status"] == "error"
    assert f"{action_dim}D" in bad["message"]
    assert len(calls) == count


def test_public_aloha_config_builds_bimanual_transforms(monkeypatch, tmp_path):
    from openpi.training import config
    c = config.get_config("aloha_eef_rtc")
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *a: None)
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *a: tr.Group())
    data = c.data.create(tmp_path, c.model)
    assert c.model.rtc_training_max_delay == 8 and c.model.action_horizon == 50
    assert c.model.action_dim == 32
    assert isinstance(data.data_transforms.inputs[0], PiperInputs)
    delta = next(t for t in data.data_transforms.inputs if isinstance(t, tr.DeltaActions))
    np.testing.assert_array_equal(delta.mask, [True]*9+[False]+[True]*9+[False])
    assert isinstance(data.data_transforms.outputs[-1], PiperOutputs)
