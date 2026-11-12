from pathlib import Path
import sys
from types import SimpleNamespace

import av
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def make_video(path, height=48, width=80):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("h264", rate=30)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((height, width, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


import convert_data_to_lerobot_fast as fast
import convert_data_to_lerobot_parallel as parallel


def test_original_resolution_and_metadata(tmp_path):
    episode = tmp_path / "episode"
    video = episode / "camera/video.mp4"
    make_video(video)
    frames = fast._decode_and_resize(video, 3, "none")
    assert len(frames) == 3
    assert all(frame.shape == (48, 80, 3) for frame in frames)
    shapes = parallel._source_camera_shapes([episode], {"camera": "image"})
    schema = SimpleNamespace(numeric_fields=[], camera_key_map={"camera": "image"})
    assert fast._build_features(schema, shapes)["image"]["shape"] == (48, 80, 3)
    assert fast._build_features(schema)["image"]["shape"] == (224, 224, 3)
    output = tmp_path / "output.mp4"
    fast._encode_video_arrays(frames, output, 30, 23, 2)
    assert all(frame.shape == (48, 80, 3) for frame in fast._decode_and_resize(output, 3, "none"))


def test_mixed_resolutions_fail(tmp_path):
    episodes = [tmp_path / "a", tmp_path / "b"]
    for episode, height in zip(episodes, (48, 60)):
        make_video(episode / "camera/video.mp4", height=height)
    with pytest.raises(ValueError, match="differs"):
        parallel._source_camera_shapes(episodes, {"camera": "image"})
