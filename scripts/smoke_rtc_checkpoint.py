"""Offline GPU check; never opens a robot connection or binds a service port."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import msgpack
import numpy as np
from openpi_zmq_server import PIZmqServer

parser=argparse.ArgumentParser()
parser.add_argument('--checkpoint',required=True)
args=parser.parse_args()
server=PIZmqServer(config_name='flexiv_plug_rtc',checkpoint_dir=args.checkpoint)
start=time.monotonic()
server._load_policy()
print(json.dumps({'loaded_seconds':time.monotonic()-start}),flush=True)
obs={'state':np.array([0.4,0,0.3,1,0,0,0,1,0,0.5],np.float32),
     'observation/image':np.zeros((224,224,3),np.uint8),
     'observation/wrist_image':np.zeros((224,224,3),np.uint8)}
noise=np.random.default_rng(1).normal(size=(50,32)).astype(np.float32)
p=server._policy
base=p.infer(obs,noise=noise)
zero=p.infer(obs,noise=noise,rtc_prefix_length=0)
assert base['actions'].shape==(50,10) and np.isfinite(base['actions']).all()
np.testing.assert_allclose(base['actions'],zero['actions'],atol=2e-4,rtol=2e-4)
print(json.dumps({'zero_prefix_matches_baseline':True,'max_diff':float(np.max(np.abs(base['actions']-zero['actions'])))}),flush=True)
for delay in (4,8):
    prefix=base['actions'][5:5+delay].copy()
    moved={**obs,'state':obs['state']+np.array([0.01,0,0,0,0,0,0,0,0,0],np.float32)}
    out=p.infer(moved,noise=noise,rtc_prefix=prefix,rtc_prefix_length=delay)
    assert out['actions'].shape==(50,10) and np.isfinite(out['actions']).all()
    np.testing.assert_array_equal(out['actions'][:delay],prefix)
    # Validate the MODEL output prefix too, without Policy's final round-trip correction.
    model_inputs=p._input_transform({**moved,'actions':np.pad(prefix,((0,50-delay),(0,0)))})
    import jax
    import jax.numpy as jnp
    from openpi.models.model import Observation
    normalized=model_inputs.pop('actions')
    batched=jax.tree.map(lambda v:jnp.asarray(v)[None],model_inputs)
    sampled=np.asarray(p._sample_actions(jax.random.key(1),Observation.from_dict(batched),
        noise=jnp.asarray(noise)[None],rtc_prefix=jnp.asarray(normalized)[None],rtc_prefix_length=jnp.array(delay)))
    np.testing.assert_array_equal(sampled[0,:delay],np.asarray(jnp.asarray(normalized))[:delay])
    print(json.dumps({'prefix_length':delay,'shape':list(out['actions'].shape),'finite':True,
                      'model_prefix_exact':True,'physical_prefix_exact':True,
                      'infer_ms':out['policy_timing']['infer_ms']}),flush=True)
ok,jpeg=cv2.imencode('.jpg',obs['observation/image']);assert ok
request={'cmd':'predict_rtc','state':obs['state'].tolist(),'observation/image':jpeg.tobytes(),
         'observation/wrist_image':jpeg.tobytes(),
         'rtc':{'prefix_length':8,'prefix_actions':base['actions'][5:13].tolist()}}
reply=server._handle_request(msgpack.packb(request))
assert reply['status']=='ok' and reply['rtc']['applied']
assert np.isfinite(reply['actions']).all()
print(json.dumps({'msgpack_request':'passed','rtc':reply['rtc'],'infer_ms':reply['infer_time_ms']}),flush=True)
print('RTC_CHECKPOINT_SMOKE_OK',flush=True)
