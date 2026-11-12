"""Direct-video raw VTLA to LeRobot v2.1 converter.

This converter is a compatibility variant of ``convert_data_to_lerobot_fast.py``.
It keeps raw numeric field names in the LeRobot dataset instead of collapsing
different robot schemas into ``observation.state`` and ``action``.

Examples of preserved numeric feature names:

    observation.state.eef_pose
    observation.state.joint_position
    actions.eef_pose
    actions.joint_position

Camera names are still normalized to LeRobot's ``observation.images.*`` naming.

This converter is intended to prepare datasets for PI training, so tactile
camera videos are excluded by default. Pass ``--include-tactile`` explicitly
when converting a dataset for a tactile-aware model or another codebase that
actually consumes tactile observations.


.venv/bin/python scripts/convert_data_to_lerobot_fast.py \
    --src /path/to/raw_episodes \
    --repo-id example_dataset

"""
from __future__ import annotations

import argparse
from collections import deque
import contextlib
import concurrent.futures
import csv
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import av
import datasets
from datasets import concatenate_datasets
import numpy as np
from PIL import Image
import tqdm

try:
    import cv2
except ImportError:  # pragma: no cover - runtime fallback for minimal envs.
    cv2 = None

from lerobot.common.datasets.compute_stats import auto_downsample_height_width
from lerobot.common.datasets.compute_stats import get_feature_stats
from lerobot.common.datasets.compute_stats import sample_indices
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import check_timestamps_sync
from lerobot.common.datasets.utils import embed_images
from lerobot.common.datasets.utils import get_episode_data_index
from lerobot.common.datasets.utils import hf_transform_to_torch

try:
    from lerobot.common.constants import HF_LEROBOT_HOME as LEROBOT_HOME
except ImportError:
    LEROBOT_HOME = Path(
        os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot")
    )


RGB_CAMERA_KEY_MAP: dict[str, str] = {
    "observation.image.third_view": "observation.images.third_view",
    "observation.image.second_third_view": "observation.images.second_third_view",
    "observation.image.left_wrist_view": "observation.images.left_wrist_view",
    "observation.image.right_wrist_view": "observation.images.right_wrist_view",
}
TACTILE_CAMERA_KEY_MAP: dict[str, str] = {
    "observation.image.left_wrist_left_tactile": "observation.images.left_wrist_left_tactile",
    "observation.image.left_wrist_right_tactile": "observation.images.left_wrist_right_tactile",
    "observation.image.right_wrist_left_tactile": "observation.images.right_wrist_left_tactile",
    "observation.image.right_wrist_right_tactile": "observation.images.right_wrist_right_tactile",
}

STATE_FIELD_CANDIDATES = (
    "observation.state.eef_pose",
    "observation.state.joint_position",
)
ACTION_FIELD_CANDIDATES = (
    "actions.eef_pose",
    "actions.joint_position",
)
ACTION_STATE_FALLBACKS = {
    "actions.eef_pose": "observation.state.eef_pose",
    "actions.joint_position": "observation.state.joint_position",
}
EEF_POSE_NAME_ALIASES = {
    "tcp.x": "x",
    "tcp.y": "y",
    "tcp.z": "z",
    "tcp.r1": "r1",
    "tcp.r2": "r2",
    "tcp.r3": "r3",
    "tcp.r4": "r4",
    "tcp.r5": "r5",
    "tcp.r6": "r6",
    "gripper.pos": "gripper",
}

TARGET_HW = (224, 224)  # (height, width)
FPS = 30


class NumericFieldSpec:
    def __init__(self, key: str, names: list[str], fallback_key: str | None = None) -> None:
        if not names:
            raise ValueError(f"{key}: expected at least one numeric column after timestamp")
        self.key = key
        self.names = names
        self.shape = (len(names),)
        self.fallback_key = fallback_key


class EpisodeSchema:
    def __init__(self, numeric_fields: list[NumericFieldSpec], camera_key_map: dict[str, str]) -> None:
        self.numeric_fields = numeric_fields
        self.camera_key_map = camera_key_map


def _selected_camera_key_map(include_tactile: bool) -> dict[str, str]:
    camera_key_map = dict(RGB_CAMERA_KEY_MAP)
    if include_tactile:
        camera_key_map.update(TACTILE_CAMERA_KEY_MAP)
    return camera_key_map


def _normalize_numeric_names(field_key: str, names: list[str]) -> list[str]:
    if field_key.endswith(".eef_pose"):
        return [EEF_POSE_NAME_ALIASES.get(name, name) for name in names]
    return names


def _read_numeric_header(csv_path: Path, field_key: str) -> list[str]:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path}: empty csv") from exc

    if len(header) < 2:
        raise ValueError(f"{csv_path}: expected timestamp plus at least one value column")
    return _normalize_numeric_names(field_key, header[1:])


def _read_numeric_csv(csv_path: Path, expected_names: list[str], field_key: str) -> np.ndarray:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path}: empty csv") from exc

        names = _normalize_numeric_names(field_key, header[1:])
        if names != expected_names:
            raise ValueError(f"{csv_path}: expected columns {expected_names}, got {header[1:]}")

        rows: list[list[float]] = []
        for row_index, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(
                    f"{csv_path}: row {row_index} has {len(row)} columns, expected {len(header)}"
                )
            rows.append([float(value) for value in row[1:]])

    return np.asarray(rows, dtype=np.float32)


class _EpisodeProbe:
    """Cache file checks and headers for one discovery pass, never across runs."""

    def __init__(self) -> None:
        self.existence: dict[Path, bool] = {}
        self.headers: dict[tuple[Path, bool], list[str]] = {}

    def exists(self, path: Path) -> bool:
        if path not in self.existence:
            self.existence[path] = path.exists()
        return self.existence[path]

    def header(self, path: Path, key: str) -> list[str]:
        cache_key = (path, key.endswith(".eef_pose"))
        if cache_key not in self.headers:
            self.headers[cache_key] = _read_numeric_header(path, key)
        return self.headers[cache_key]


def _inspect_episode_schema(
    ep_dir: Path,
    include_tactile: bool,
    strict_cameras: bool = False,
    probe: _EpisodeProbe | None = None,
) -> EpisodeSchema:
    probe = probe or _EpisodeProbe()
    numeric_fields: list[NumericFieldSpec] = []
    state_fields: dict[str, NumericFieldSpec] = {}
    for key in STATE_FIELD_CANDIDATES:
        csv_path = ep_dir / key / "data.csv"
        if probe.exists(csv_path):
            field = NumericFieldSpec(key=key, names=probe.header(csv_path, key))
            numeric_fields.append(field)
            state_fields[key] = field

    if not state_fields:
        raise ValueError("missing compatible observation.state.* csv")

    missing_actions: list[str] = []
    for action_key in ACTION_FIELD_CANDIDATES:
        csv_path = ep_dir / action_key / "data.csv"
        fallback_key = ACTION_STATE_FALLBACKS[action_key]
        if probe.exists(csv_path):
            names = probe.header(csv_path, action_key)
        else:
            fallback_field = state_fields.get(fallback_key)
            if fallback_field is None:
                missing_actions.append(f"{action_key} or fallback {fallback_key}")
                continue
            names = fallback_field.names
        numeric_fields.append(NumericFieldSpec(key=action_key, names=names, fallback_key=fallback_key))

    if missing_actions:
        raise ValueError("missing compatible action csv(s): " + ", ".join(missing_actions))

    selected_camera_key_map = _selected_camera_key_map(include_tactile)
    camera_key_map = {
        input_key: output_key
        for input_key, output_key in selected_camera_key_map.items()
        if probe.exists(ep_dir / input_key / "video.mp4")
    }
    missing_cameras = sorted(set(selected_camera_key_map) - set(camera_key_map))
    if strict_cameras and missing_cameras:
        raise FileNotFoundError(f"missing camera video(s): {', '.join(missing_cameras)}")
    if not camera_key_map:
        raise FileNotFoundError("missing all compatible camera videos")

    return EpisodeSchema(numeric_fields=numeric_fields, camera_key_map=camera_key_map)


def _read_numeric_episode_fields(
    ep_dir: Path,
    numeric_fields: list[NumericFieldSpec],
) -> dict[str, np.ndarray]:
    arrays = {}
    for field in numeric_fields:
        csv_path = ep_dir / field.key / "data.csv"
        if not csv_path.exists() and field.fallback_key is not None:
            csv_path = ep_dir / field.fallback_key / "data.csv"
        arrays[field.key] = _read_numeric_csv(csv_path, field.names, field.key)
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{ep_dir}: numeric fields have different row counts: {lengths}")
    return arrays


def _resolve_episode_task(ep_dir: Path, manual_task: str | None) -> str:
    if manual_task is not None:
        task = manual_task.strip()
        if not task:
            raise ValueError("--task was provided but is empty")
        return task

    metadata_path = ep_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path}: missing metadata.json and --task was not provided")

    with metadata_path.open() as f:
        metadata = json.load(f)

    task = str(metadata.get("language_instruction", "")).strip()
    if not task:
        raise ValueError(f"{metadata_path}: missing non-empty language_instruction and --task was not provided")
    return task


def _episode_matches_schema(
    ep_dir: Path, schema: EpisodeSchema, probe: _EpisodeProbe | None = None
) -> tuple[bool, str]:
    probe = probe or _EpisodeProbe()
    for field in schema.numeric_fields:
        csv_path = ep_dir / field.key / "data.csv"
        if not probe.exists(csv_path):
            if field.fallback_key is None:
                return False, f"missing {field.key}/data.csv"
            csv_path = ep_dir / field.fallback_key / "data.csv"
            if not probe.exists(csv_path):
                return False, f"missing {field.key}/data.csv and fallback {field.fallback_key}/data.csv"
        names = probe.header(csv_path, field.key)
        if names != field.names:
            return False, f"{field.key} columns differ: expected {field.names}, got {names}"

    for input_key in schema.camera_key_map:
        if not probe.exists(ep_dir / input_key / "video.mp4"):
            return False, f"missing {input_key}/video.mp4"

    return True, ""


def _schemas_equal(left: EpisodeSchema, right: EpisodeSchema) -> bool:
    left_numeric = [(field.key, field.names) for field in left.numeric_fields]
    right_numeric = [(field.key, field.names) for field in right.numeric_fields]
    return left_numeric == right_numeric and left.camera_key_map == right.camera_key_map


def _resize_rgb(arr: np.ndarray, backend: str) -> np.ndarray:
    if backend == "none" or arr.shape[:2] == TARGET_HW:
        return np.asarray(arr, dtype=np.uint8)

    size = (TARGET_HW[1], TARGET_HW[0])
    if backend == "cv2":
        if cv2 is None:
            raise RuntimeError("OpenCV is not available; rerun with --resize-backend pil")
        return cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)

    if backend == "pil":
        return np.asarray(Image.fromarray(arr).resize(size, Image.BILINEAR), dtype=np.uint8)

    raise ValueError(f"Unsupported resize backend: {backend}")


def _decode_and_resize(vid_path: Path, expected: int, resize_backend: str) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with av.open(str(vid_path)) as container:
        for frame in container.decode(video=0):
            if len(frames) >= expected:
                break
            arr = frame.to_ndarray(format="rgb24")
            frames.append(_resize_rgb(arr, resize_backend))

    if not frames:
        raise RuntimeError(f"{vid_path}: decoded 0 frames")

    while len(frames) < expected:
        frames.append(frames[-1])
    return frames


def _effective_worker_count(requested: int | None, item_count: int, default_cap: int) -> int:
    if item_count <= 1:
        return 1
    if requested is None:
        return max(1, min(item_count, default_cap))
    if requested < 1:
        raise ValueError("worker count must be >= 1")
    return min(requested, item_count)


def _decode_episode_cameras(
    ep_dir: Path,
    camera_key_map: dict[str, str],
    expected_frames: int,
    video_workers: int | None,
    resize_backend: str,
) -> dict[str, list[np.ndarray]]:
    missing = [
        input_key
        for input_key in camera_key_map
        if not (ep_dir / input_key / "video.mp4").exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing camera video(s): {', '.join(missing)}")

    workers = _effective_worker_count(video_workers, len(camera_key_map), default_cap=4)
    results: dict[str, list[np.ndarray]] = {}

    def decode_one(input_key: str, output_key: str) -> tuple[str, list[np.ndarray]]:
        return output_key, _decode_and_resize(ep_dir / input_key / "video.mp4", expected_frames, resize_backend)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(decode_one, input_key, output_key)
            for input_key, output_key in camera_key_map.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            output_key, frames = future.result()
            results[output_key] = frames

    return results


def _encode_video_arrays(
    frames: list[np.ndarray],
    video_path: Path,
    fps: int,
    crf: int,
    gop: int | None,
) -> None:
    if not frames:
        raise ValueError(f"{video_path}: no frames to encode")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    options = {"crf": str(crf)}
    if gop is not None:
        options["g"] = str(gop)

    av.logging.set_level(av.logging.ERROR)
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream("h264", rate=fps, options=options)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for arr in frames:
            if arr.shape[:2] != (height, width):
                raise ValueError(f"{video_path}: frame shape changed from {(height, width)} to {arr.shape[:2]}")
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)

        for packet in stream.encode():
            output.mux(packet)

    if not video_path.exists():
        raise OSError(f"Video encoding did not create {video_path}")


def _compute_video_stats_from_frames(frames: list[np.ndarray]) -> dict[str, np.ndarray]:
    sampled = sample_indices(len(frames))
    images = None
    for i, idx in enumerate(sampled):
        img = np.transpose(frames[idx], (2, 0, 1)).astype(np.uint8, copy=False)
        img = auto_downsample_height_width(img)
        if images is None:
            images = np.empty((len(sampled), *img.shape), dtype=np.uint8)
        images[i] = img

    stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
    return {key: value if key == "count" else np.squeeze(value / 255.0, axis=0) for key, value in stats.items()}


def _compute_episode_stats_direct(
    episode_buffer: dict[str, np.ndarray],
    cam_frames: dict[str, list[np.ndarray]],
    features: dict,
) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for key, ft in features.items():
        if ft["dtype"] == "string":
            continue
        if ft["dtype"] in ["image", "video"]:
            stats[key] = _compute_video_stats_from_frames(cam_frames[key])
            continue

        data = np.asarray(episode_buffer[key])
        stats[key] = get_feature_stats(data, axis=0, keepdims=data.ndim == 1)
    return stats


def _save_episode_table(dataset: LeRobotDataset, episode_buffer: dict, episode_index: int) -> None:
    episode_dict = {key: episode_buffer[key] for key in dataset.hf_features}
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=dataset.hf_features, split="train")
    ep_dataset = embed_images(ep_dataset)
    dataset.hf_dataset = concatenate_datasets([dataset.hf_dataset, ep_dataset])
    dataset.hf_dataset.set_transform(hf_transform_to_torch)
    ep_data_path = dataset.root / dataset.meta.get_data_file_path(ep_index=episode_index)
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)


def _save_episode_direct(
    dataset: LeRobotDataset,
    numeric_arrays: dict[str, np.ndarray],
    cam_frames: dict[str, list[np.ndarray]],
    camera_output_keys: list[str],
    task: str,
    video_encode_workers: int,
    video_crf: int,
    video_gop: int | None,
) -> tuple[float, float, float]:
    episode_index = dataset.meta.total_episodes
    episode_length = len(next(iter(numeric_arrays.values())))
    episode_tasks = list({task})

    for episode_task in episode_tasks:
        if dataset.meta.get_task_index(episode_task) is None:
            dataset.meta.add_task(episode_task)

    timestamps = np.arange(episode_length, dtype=np.float32) / dataset.fps
    episode_indices = np.full((episode_length,), episode_index, dtype=np.int64)
    episode_buffer = dict(numeric_arrays)
    episode_buffer.update(
        {
            "timestamp": timestamps,
            "frame_index": np.arange(episode_length, dtype=np.int64),
            "episode_index": episode_indices,
            "index": np.arange(dataset.meta.total_frames, dataset.meta.total_frames + episode_length, dtype=np.int64),
            "task_index": np.full(
                (episode_length,),
                dataset.meta.get_task_index(task),
                dtype=np.int64,
            ),
        }
    )

    table_t0 = time.perf_counter()
    _save_episode_table(dataset, episode_buffer, episode_index)
    table_s = time.perf_counter() - table_t0

    stats_t0 = time.perf_counter()
    ep_stats = _compute_episode_stats_direct(episode_buffer, cam_frames, dataset.features)
    stats_s = time.perf_counter() - stats_t0

    encode_t0 = time.perf_counter()
    workers = _effective_worker_count(video_encode_workers, len(camera_output_keys), default_cap=4)

    def encode_one(output_key: str) -> None:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, output_key)
        _encode_video_arrays(
            cam_frames[output_key],
            video_path,
            fps=dataset.fps,
            crf=video_crf,
            gop=video_gop,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(encode_one, camera_output_keys))
    encode_s = time.perf_counter() - encode_t0

    dataset.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

    ep_data_index = get_episode_data_index(dataset.meta.episodes, [episode_index])
    ep_data_index_np = {key: tensor.numpy() for key, tensor in ep_data_index.items()}
    check_timestamps_sync(
        timestamps,
        episode_indices,
        ep_data_index_np,
        dataset.fps,
        dataset.tolerance_s,
    )

    # Check only this episode here; do one complete inventory after conversion.
    expected_files = [dataset.root / dataset.meta.get_data_file_path(ep_index=episode_index)]
    expected_files.extend(
        dataset.root / dataset.meta.get_video_file_path(episode_index, key) for key in camera_output_keys
    )
    for path in expected_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Missing or empty episode output: {path}")

    return table_s, stats_s, encode_s


def _find_episode_dirs(src: Path) -> list[Path]:
    """Find episodes up to three levels deep, without scanning inside episodes."""
    episodes: set[Path] = set()
    pending = [(src, 0)]
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                is_episode = entry.name.startswith("episode_")
                # At the deepest supported level, only episode directories matter.
                if not is_episode and depth >= 2:
                    continue
                if not entry.is_dir():
                    continue
                path = directory / entry.name
                if is_episode:
                    episodes.add(path)
                else:
                    pending.append((path, depth + 1))
    return sorted(episodes)


def _discover_episodes(
    src: Path,
    limit: int | None,
    include_tactile: bool,
    strict_cameras: bool,
    scan_workers: int = 8,
) -> tuple[list[Path], EpisodeSchema, list[tuple[Path, str]]]:
    if scan_workers < 1:
        raise ValueError("scan_workers must be >= 1")
    print(f"Finding episode directories under {src} ...", flush=True)
    all_eps = _find_episode_dirs(src)
    print(f"Found {len(all_eps)} episode directories; checking with {scan_workers} workers.", flush=True)
    kept: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    schema: EpisodeSchema | None = None

    def inspect(ep: Path):
        probe = _EpisodeProbe()
        try:
            if not ep.is_dir():
                return None, probe, None
            return _inspect_episode_schema(ep, include_tactile, strict_cameras, probe), probe, None
        except Exception as exc:
            return None, probe, f"{type(exc).__name__}: {exc}"

    # Keep results in original sorted order: the first schema, --limit, episode
    # indices and task assignment must not depend on which worker finishes first.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=scan_workers)
    pending = deque()
    remaining = iter(all_eps)
    try:
        for ep in remaining:
            pending.append((ep, executor.submit(inspect, ep)))
            if len(pending) >= scan_workers * 2:
                break
        with tqdm.tqdm(total=len(all_eps), desc="Checking episodes", unit="episode") as progress:
            while pending:
                ep, future = pending.popleft()
                ep_schema, probe, error = future.result()
                try:
                    if error is not None:
                        skipped.append((ep, error))
                    elif ep_schema is not None:
                        if schema is None:
                            schema = ep_schema
                        # One compatibility check, sharing the inspection cache.
                        ok, why = _episode_matches_schema(ep, schema, probe)
                        if ok:
                            kept.append(ep)
                        else:
                            skipped.append((ep, why))
                except Exception as exc:
                    skipped.append((ep, f"{type(exc).__name__}: {exc}"))
                progress.update(1)
                if limit is not None and len(kept) >= limit:
                    break
                ep = next(remaining, None)
                if ep is not None:
                    pending.append((ep, executor.submit(inspect, ep)))
    finally:
        for _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    if schema is None:
        raise SystemExit(f"No compatible episodes under {src}")
    print(f"Scanned {len(all_eps)} raw episodes, kept {len(kept)} with compatible schema.", flush=True)
    return kept, schema, skipped


@contextlib.contextmanager
def _local_episode(
    ep_dir: Path, schema: EpisodeSchema, cache_dir: Path | None, task: str | None, copy_workers: int
):
    """Cache only inputs consumed by conversion; release space after each episode."""
    if cache_dir is None:
        yield ep_dir
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="episode-", dir=cache_dir) as temporary:
        local = Path(temporary)
        files: set[Path] = set()
        for field in schema.numeric_fields:
            path = ep_dir / field.key / "data.csv"
            if not path.exists() and field.fallback_key is not None:
                path = ep_dir / field.fallback_key / "data.csv"
            files.add(path)
        files.update(ep_dir / key / "video.mp4" for key in schema.camera_key_map)
        if task is None:
            files.add(ep_dir / "metadata.json")

        def copy(path: Path) -> None:
            target = local / path.relative_to(ep_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if target.stat().st_size != path.stat().st_size:
                raise OSError(f"Incomplete local copy: {path}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=copy_workers) as executor:
            list(executor.map(copy, sorted(files)))
        yield local


def _validate_dataset_files(dataset: LeRobotDataset) -> None:
    """Retain the original global count checks, but run them only once."""
    print("Checking final dataset file counts ...", flush=True)
    video_count = 0
    parquet_count = 0
    for _, _, files in os.walk(dataset.root):
        video_count += sum(name.endswith(".mp4") for name in files)
        parquet_count += sum(name.endswith(".parquet") for name in files)
    expected_videos = dataset.num_episodes * len(dataset.meta.video_keys)
    if video_count != expected_videos or parquet_count != dataset.num_episodes:
        raise RuntimeError(
            f"Dataset output count mismatch: videos {video_count}/{expected_videos}, "
            f"parquet {parquet_count}/{dataset.num_episodes}"
        )


def _build_features(schema: EpisodeSchema, camera_shapes: dict | None = None) -> dict:
    features: dict = {}
    for field in schema.numeric_fields:
        features[field.key] = {
            "dtype": "float32",
            "shape": field.shape,
            "names": [field.names],
        }
    for output_key in schema.camera_key_map.values():
        features[output_key] = {
            "dtype": "video",
            "shape": camera_shapes[output_key] if camera_shapes is not None else (TARGET_HW[0], TARGET_HW[1], 3),
            "names": ["height", "width", "channels"],
        }
    return features


def _infer_robot_type(schema: EpisodeSchema, include_tactile: bool) -> str:
    all_names = [name for field in schema.numeric_fields for name in field.names]
    has_left = any(name.startswith("left_") for name in all_names)
    has_right = any(name.startswith("right_") for name in all_names)
    has_right_camera = any("right_wrist" in key for key in schema.camera_key_map)
    arm_type = "dual" if has_right or (has_left and has_right_camera) else "single"
    suffix = "_tactile" if include_tactile and any("tactile" in key for key in schema.camera_key_map) else ""
    return f"vtla_{arm_type}{suffix}"


def convert(
    src: Path,
    repo_id: str,
    task: str | None,
    limit: int | None,
    include_tactile: bool,
    strict_cameras: bool,
    video_workers: int | None,
    video_encode_workers: int,
    resize_backend: str,
    video_crf: int,
    video_gop: int | None,
    robot_type: str | None,
    profile: bool,
    scan_workers: int = 8,
    local_cache_dir: Path | None = None,
    copy_workers: int = 8,
) -> None:
    src = Path(src)
    if copy_workers < 1:
        raise ValueError("copy_workers must be >= 1")
    episodes, schema, discovery_skipped = _discover_episodes(
        src=src,
        limit=limit,
        include_tactile=include_tactile,
        strict_cameras=strict_cameras,
        scan_workers=scan_workers,
    )
    if not episodes:
        raise SystemExit(f"No episodes under {src}")

    camera_output_keys = list(schema.camera_key_map.values())
    decode_workers = _effective_worker_count(video_workers, len(schema.camera_key_map), default_cap=4)
    encode_workers = _effective_worker_count(video_encode_workers, len(schema.camera_key_map), default_cap=4)
    robot_type = robot_type or _infer_robot_type(schema, include_tactile)

    print(f"Found {len(episodes)} episodes; direct-video converting with raw numeric fields:")
    for field in schema.numeric_fields:
        print(f"  - {field.key}: shape={field.shape}, names={field.names}")
    print("Cameras:")
    for input_key, output_key in schema.camera_key_map.items():
        print(f"  - {input_key} -> {output_key}")
    print(
        "Acceleration: "
        f"decode_workers={decode_workers}, "
        f"encode_workers={encode_workers}, "
        f"resize_backend={resize_backend}, "
        f"video_crf={video_crf}, "
        f"video_gop={video_gop}, "
        f"robot_type={robot_type}"
    )

    ds_path = LEROBOT_HOME / repo_id
    if ds_path.exists():
        print(f"Removing existing dataset at {ds_path}")
        shutil.rmtree(ds_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type=robot_type,
        features=_build_features(schema),
        use_videos=True,
        tolerance_s=0.1,
    )

    skipped: list[tuple[Path, str]] = list(discovery_skipped)
    converted = 0
    timings = {
        "read_numeric": 0.0,
        "decode": 0.0,
        "table": 0.0,
        "stats": 0.0,
        "encode": 0.0,
        "total": 0.0,
    }

    for ep_dir in tqdm.tqdm(episodes, desc="Direct converting"):
        episode_t0 = time.perf_counter()
        try:
            with _local_episode(ep_dir, schema, local_cache_dir, task, copy_workers) as input_dir:
                numeric_t0 = time.perf_counter()
                numeric_arrays = _read_numeric_episode_fields(input_dir, schema.numeric_fields)
                n = len(next(iter(numeric_arrays.values())))
                if n < 2:
                    skipped.append((ep_dir, f"only {n} frames"))
                    continue
                read_numeric_s = time.perf_counter() - numeric_t0
                timings["read_numeric"] += read_numeric_s
                episode_task = _resolve_episode_task(input_dir, task)

                decode_t0 = time.perf_counter()
                cam_frames = _decode_episode_cameras(
                    ep_dir=input_dir,
                    camera_key_map=schema.camera_key_map,
                    expected_frames=n,
                    video_workers=decode_workers,
                    resize_backend=resize_backend,
                )
                decode_s = time.perf_counter() - decode_t0
                timings["decode"] += decode_s

                table_s, stats_s, encode_s = _save_episode_direct(
                    dataset=dataset,
                    numeric_arrays=numeric_arrays,
                    cam_frames=cam_frames,
                    camera_output_keys=camera_output_keys,
                    task=episode_task,
                    video_encode_workers=encode_workers,
                    video_crf=video_crf,
                    video_gop=video_gop,
                )
                timings["table"] += table_s
                timings["stats"] += stats_s
                timings["encode"] += encode_s

                converted += 1
                episode_s = time.perf_counter() - episode_t0
                timings["total"] += episode_s
                if profile:
                    tqdm.tqdm.write(
                        f"{ep_dir.name}: frames={n} "
                        f"numeric={read_numeric_s:.2f}s decode={decode_s:.2f}s "
                        f"table={table_s:.2f}s stats={stats_s:.2f}s "
                        f"encode={encode_s:.2f}s total={episode_s:.2f}s"
                    )
        except Exception as e:
            if isinstance(e, OSError) and e.errno == errno.ENOSPC:
                raise RuntimeError("Conversion disk is full; local work is retained by the wrapper") from e
            skipped.append((ep_dir, f"{type(e).__name__}: {e}"))
            continue

    _validate_dataset_files(dataset)
    print(f"\nDataset written to {ds_path}")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {len(skipped)}")
    if converted:
        print("Timing totals:")
        for name in ("read_numeric", "decode", "table", "stats", "encode", "total"):
            print(f"  {name:12s}: {timings[name]:.2f}s")
    for ep, why in skipped[:20]:
        print(f"    - {ep.name}: {why}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="Root of raw VTLA episodes")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override metadata.json language_instruction for every episode",
    )
    parser.add_argument("--limit", type=int, default=None, help="Convert first N valid episodes")
    tactile_group = parser.add_mutually_exclusive_group()
    tactile_group.add_argument(
        "--include-tactile",
        dest="include_tactile",
        action="store_true",
        help=(
            "Include tactile camera videos. This converter targets PI training, which does not consume tactile "
            "observations by default; enable this only for a tactile-aware model or codebase."
        ),
    )
    tactile_group.add_argument(
        "--no-tactile",
        dest="include_tactile",
        action="store_false",
        help="Deprecated compatibility flag; tactile cameras are already excluded by default.",
    )
    parser.set_defaults(include_tactile=False)
    parser.add_argument(
        "--strict-cameras",
        action="store_true",
        help="Require every known RGB/tactile camera to exist instead of using cameras present in the first episode",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=8,
        help="Parallel camera decode workers per episode; default min(num_cameras, 4)",
    )
    parser.add_argument(
        "--video-encode-workers",
        type=int,
        default=8,
        help="Parallel direct MP4 encode workers per episode; default min(num_cameras, 4)",
    )
    parser.add_argument("--resize-backend", choices=("cv2", "pil"), default="cv2")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-gop", type=int, default=2)
    parser.add_argument("--robot-type", type=str, default=None, help="Override inferred LeRobot robot_type")
    parser.add_argument("--profile", action="store_true", help="Print per-episode timing breakdown")
    parser.add_argument("--scan-workers", type=int, default=8, help="Bounded parallel episode checks")
    parser.add_argument("--local-cache-dir", type=Path, default=None, help="Stage one episode at a time on local disk")
    parser.add_argument("--copy-workers", type=int, default=8, help="Parallel local input copies")
    args = parser.parse_args()

    if args.include_tactile:
        print(
            "Tactile conversion enabled explicitly. Use this only when the target model/codebase consumes tactile "
            "observations; standard PI training does not."
        )
    else:
        print(
            "PI training conversion: tactile camera videos are excluded by default. "
            "Pass --include-tactile only for a tactile-aware model or codebase."
        )

    convert(
        src=args.src,
        repo_id=args.repo_id,
        task=args.task,
        limit=args.limit,
        include_tactile=args.include_tactile,
        strict_cameras=args.strict_cameras,
        video_workers=args.video_workers,
        video_encode_workers=args.video_encode_workers,
        resize_backend=args.resize_backend,
        video_crf=args.video_crf,
        video_gop=args.video_gop,
        robot_type=args.robot_type,
        profile=args.profile,
        scan_workers=args.scan_workers,
        local_cache_dir=args.local_cache_dir,
        copy_workers=args.copy_workers,
    )


if __name__ == "__main__":
    main()
