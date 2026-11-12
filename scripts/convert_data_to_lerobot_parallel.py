"""Episode-parallel raw VTLA to LeRobot v2.1 converter.

Images retain their source resolution; resizing/padding belongs to training.
Video streams are re-encoded, so this is not a byte-for-byte video copy.

This script keeps the output format from ``convert_data_to_lerobot_fast.py`` but
parallelizes expensive per-episode work. Worker processes write episode parquet
and video files to deterministic paths; the main process updates LeRobot
metadata in episode order to avoid concurrent metadata mutation.

This converter is intended to prepare datasets for PI training, so tactile
camera videos are excluded by default. Pass ``--include-tactile`` explicitly
when converting a dataset for a tactile-aware model or another codebase that
actually consumes tactile observations.


.venv/bin/python scripts/convert_data_to_lerobot_parallel.py \
    --src /path/to/raw_episodes \
    --repo-id example_dataset \
    --episode-workers 8

"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import time
from typing import Any

import convert_data_to_lerobot_fast as fast
import datasets
from lerobot.common.datasets.utils import check_timestamps_sync
from lerobot.common.datasets.utils import get_episode_data_index
import numpy as np
import tqdm


@dataclass(frozen=True)
class EpisodeJob:
    ep_dir: str
    episode_index: int
    start_frame_index: int
    episode_length: int
    task: str
    task_index: int
    numeric_fields: list[Any]
    camera_key_map: dict[str, str]
    camera_output_keys: list[str]
    root: str
    data_file_path: str
    video_file_paths: dict[str, str]
    hf_features: Any
    features: dict[str, Any]
    fps: int
    video_workers: int | None
    video_encode_workers: int
    resize_backend: str
    video_crf: int
    video_gop: int | None


@dataclass(frozen=True)
class EpisodeResult:
    episode_index: int
    episode_length: int
    task: str
    stats: dict[str, dict]
    timings: dict[str, float]


def _source_camera_shapes(episodes: list[Path], camera_key_map: dict[str, str]) -> dict[str, tuple[int, int, int]]:
    """Check source sizes before creating output; each camera has a fixed dataset shape."""
    shapes = {}
    for ep_dir in tqdm.tqdm(episodes, desc="Checking source video sizes"):
        for input_key, output_key in camera_key_map.items():
            path = ep_dir / input_key / "video.mp4"
            with fast.av.open(str(path)) as container:
                stream = container.streams.video[0]
                shape = (stream.height, stream.width, 3)
            if min(shape[:2]) <= 0 or any(size % 2 for size in shape[:2]):
                raise ValueError(f"{path}: source size {shape} cannot be encoded as yuv420p without resizing")
            if output_key in shapes and shapes[output_key] != shape:
                raise ValueError(f"{path}: camera size {shape} differs from {shapes[output_key]}; split datasets by resolution")
            shapes[output_key] = shape
    return shapes


def _numeric_csv_path(ep_dir: Path, field: Any) -> Path:
    csv_path = ep_dir / field.key / "data.csv"
    if not csv_path.exists() and field.fallback_key is not None:
        csv_path = ep_dir / field.fallback_key / "data.csv"
    return csv_path


def _count_numeric_csv_rows(csv_path: Path) -> int:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path}: empty csv") from exc
        return sum(1 for _ in reader)


def _episode_numeric_length(ep_dir: Path, numeric_fields: list[Any]) -> int:
    lengths = {
        field.key: _count_numeric_csv_rows(_numeric_csv_path(ep_dir, field))
        for field in numeric_fields
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{ep_dir}: numeric fields have different row counts: {lengths}")
    return next(iter(lengths.values()))


def _save_episode_table_direct(
    root: Path,
    data_file_path: str,
    episode_buffer: dict[str, np.ndarray],
    hf_features: Any,
) -> None:
    episode_dict = {key: episode_buffer[key] for key in hf_features}
    ep_dataset = datasets.Dataset.from_dict(episode_dict, features=hf_features, split="train")
    ep_dataset = fast.embed_images(ep_dataset)
    ep_data_path = root / data_file_path
    ep_data_path.parent.mkdir(parents=True, exist_ok=True)
    ep_dataset.to_parquet(ep_data_path)


def _process_episode_job(job: EpisodeJob) -> EpisodeResult:
    ep_dir = Path(job.ep_dir)
    root = Path(job.root)
    timings = {
        "read_numeric": 0.0,
        "decode": 0.0,
        "table": 0.0,
        "stats": 0.0,
        "encode": 0.0,
        "total": 0.0,
    }
    episode_t0 = time.perf_counter()

    numeric_t0 = time.perf_counter()
    numeric_arrays = fast._read_numeric_episode_fields(ep_dir, job.numeric_fields)
    episode_length = len(next(iter(numeric_arrays.values())))
    if episode_length != job.episode_length:
        raise ValueError(
            f"{ep_dir}: expected {job.episode_length} frames from preflight, got {episode_length}"
        )
    timings["read_numeric"] = time.perf_counter() - numeric_t0

    decode_t0 = time.perf_counter()
    cam_frames = fast._decode_episode_cameras(
        ep_dir=ep_dir,
        camera_key_map=job.camera_key_map,
        expected_frames=episode_length,
        video_workers=job.video_workers,
        resize_backend=job.resize_backend,
    )
    for key, frames in cam_frames.items():
        expected_shape = tuple(job.features[key]["shape"])
        if any(frame.shape != expected_shape for frame in frames):
            raise ValueError(f"{ep_dir}: {key} decoded shape differs from {expected_shape}")
    timings["decode"] = time.perf_counter() - decode_t0

    timestamps = np.arange(episode_length, dtype=np.float32) / job.fps
    episode_indices = np.full((episode_length,), job.episode_index, dtype=np.int64)
    episode_buffer = dict(numeric_arrays)
    episode_buffer.update(
        {
            "timestamp": timestamps,
            "frame_index": np.arange(episode_length, dtype=np.int64),
            "episode_index": episode_indices,
            "index": np.arange(
                job.start_frame_index,
                job.start_frame_index + episode_length,
                dtype=np.int64,
            ),
            "task_index": np.full((episode_length,), job.task_index, dtype=np.int64),
        }
    )

    table_t0 = time.perf_counter()
    _save_episode_table_direct(root, job.data_file_path, episode_buffer, job.hf_features)
    timings["table"] = time.perf_counter() - table_t0

    stats_t0 = time.perf_counter()
    ep_stats = fast._compute_episode_stats_direct(episode_buffer, cam_frames, job.features)
    timings["stats"] = time.perf_counter() - stats_t0

    encode_t0 = time.perf_counter()
    encode_workers = fast._effective_worker_count(
        job.video_encode_workers,
        len(job.camera_output_keys),
        default_cap=4,
    )

    def encode_one(output_key: str) -> None:
        fast._encode_video_arrays(
            cam_frames[output_key],
            root / job.video_file_paths[output_key],
            fps=job.fps,
            crf=job.video_crf,
            gop=job.video_gop,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=encode_workers) as executor:
        list(executor.map(encode_one, job.camera_output_keys))
    timings["encode"] = time.perf_counter() - encode_t0
    timings["total"] = time.perf_counter() - episode_t0

    return EpisodeResult(
        episode_index=job.episode_index,
        episode_length=episode_length,
        task=job.task,
        stats=ep_stats,
        timings=timings,
    )


def _run_episode_jobs(jobs: list[EpisodeJob], episode_workers: int) -> dict[int, EpisodeResult]:
    if episode_workers < 1:
        raise ValueError("episode_workers must be >= 1")

    results: dict[int, EpisodeResult] = {}
    if episode_workers == 1:
        for job in tqdm.tqdm(jobs, desc="Parallel converting"):
            result = _process_episode_job(job)
            results[result.episode_index] = result
        return results

    workers = min(episode_workers, len(jobs))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_episode_job, job): job for job in jobs}
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Parallel converting",
        ):
            result = future.result()
            results[result.episode_index] = result
    return results


def _build_jobs(
    episodes: list[Path],
    dataset: Any,
    schema: Any,
    camera_output_keys: list[str],
    task: str | None,
    video_workers: int | None,
    video_encode_workers: int,
    resize_backend: str,
    video_crf: int,
    video_gop: int | None,
    skipped: list[tuple[Path, str]],
) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []
    next_frame_index = 0

    for ep_dir in episodes:
        try:
            episode_length = _episode_numeric_length(ep_dir, schema.numeric_fields)
            if episode_length < 2:
                skipped.append((ep_dir, f"only {episode_length} frames"))
                continue
            episode_task = fast._resolve_episode_task(ep_dir, task)
            if dataset.meta.get_task_index(episode_task) is None:
                dataset.meta.add_task(episode_task)
            task_index = dataset.meta.get_task_index(episode_task)
            if task_index is None:
                raise RuntimeError(f"failed to register task: {episode_task}")

            episode_index = len(jobs)
            jobs.append(
                EpisodeJob(
                    ep_dir=str(ep_dir),
                    episode_index=episode_index,
                    start_frame_index=next_frame_index,
                    episode_length=episode_length,
                    task=episode_task,
                    task_index=task_index,
                    numeric_fields=schema.numeric_fields,
                    camera_key_map=schema.camera_key_map,
                    camera_output_keys=camera_output_keys,
                    root=str(dataset.root),
                    data_file_path=str(dataset.meta.get_data_file_path(ep_index=episode_index)),
                    video_file_paths={
                        output_key: str(dataset.meta.get_video_file_path(episode_index, output_key))
                        for output_key in camera_output_keys
                    },
                    hf_features=dataset.hf_features,
                    features=dataset.features,
                    fps=dataset.fps,
                    video_workers=video_workers,
                    video_encode_workers=video_encode_workers,
                    resize_backend=resize_backend,
                    video_crf=video_crf,
                    video_gop=video_gop,
                )
            )
            next_frame_index += episode_length
        except Exception as exc:
            skipped.append((ep_dir, f"{type(exc).__name__}: {exc}"))
            continue

    return jobs


def _finalize_metadata(dataset: Any, jobs: list[EpisodeJob], results: dict[int, EpisodeResult]) -> dict[str, float]:
    timings = {
        "read_numeric": 0.0,
        "decode": 0.0,
        "table": 0.0,
        "stats": 0.0,
        "encode": 0.0,
        "total": 0.0,
    }
    for job in jobs:
        result = results[job.episode_index]
        dataset.meta.save_episode(
            result.episode_index,
            result.episode_length,
            [result.task],
            result.stats,
        )
        timestamps = np.arange(result.episode_length, dtype=np.float32) / dataset.fps
        episode_indices = np.full((result.episode_length,), result.episode_index, dtype=np.int64)
        ep_data_index = get_episode_data_index(dataset.meta.episodes, [result.episode_index])
        ep_data_index_np = {key: tensor.numpy() for key, tensor in ep_data_index.items()}
        check_timestamps_sync(
            timestamps,
            episode_indices,
            ep_data_index_np,
            dataset.fps,
            dataset.tolerance_s,
        )
        for name, value in result.timings.items():
            timings[name] += value
    return timings


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
    episode_workers: int,
) -> None:
    src = Path(src)
    episodes, schema, discovery_skipped = fast._discover_episodes(
        src=src,
        limit=limit,
        include_tactile=include_tactile,
        strict_cameras=strict_cameras,
    )
    if not episodes:
        raise SystemExit(f"No episodes under {src}")

    # Accept the old CLI flag but never resize during format conversion.
    if resize_backend != "none":
        print("--resize-backend is deprecated and ignored; preserving source resolution.")
    resize_backend = "none"
    camera_shapes = _source_camera_shapes(episodes, schema.camera_key_map)
    camera_output_keys = list(schema.camera_key_map.values())
    decode_workers = fast._effective_worker_count(video_workers, len(schema.camera_key_map), default_cap=4)
    encode_workers = fast._effective_worker_count(video_encode_workers, len(schema.camera_key_map), default_cap=4)
    robot_type = robot_type or fast._infer_robot_type(schema, include_tactile)

    print(f"Found {len(episodes)} episodes; episode-parallel converting with raw numeric fields:")
    for field in schema.numeric_fields:
        print(f"  - {field.key}: shape={field.shape}, names={field.names}")
    print("Cameras:")
    for input_key, output_key in schema.camera_key_map.items():
        print(f"  - {input_key} -> {output_key}")
    print(
        "Acceleration: "
        f"episode_workers={episode_workers}, "
        f"decode_workers={decode_workers}, "
        f"encode_workers={encode_workers}, "
        f"resize_backend={resize_backend}, "
        f"video_crf={video_crf}, "
        f"video_gop={video_gop}, "
        f"robot_type={robot_type}"
    )

    ds_path = fast.LEROBOT_HOME / repo_id
    if ds_path.exists():
        print(f"Removing existing dataset at {ds_path}")
        shutil.rmtree(ds_path)

    dataset = fast.LeRobotDataset.create(
        repo_id=repo_id,
        root=ds_path,
        fps=fast.FPS,
        robot_type=robot_type,
        features=fast._build_features(schema, camera_shapes=camera_shapes),
        use_videos=True,
        tolerance_s=0.1,
    )

    skipped: list[tuple[Path, str]] = list(discovery_skipped)
    jobs = _build_jobs(
        episodes=episodes,
        dataset=dataset,
        schema=schema,
        camera_output_keys=camera_output_keys,
        task=task,
        video_workers=video_workers,
        video_encode_workers=video_encode_workers,
        resize_backend=resize_backend,
        video_crf=video_crf,
        video_gop=video_gop,
        skipped=skipped,
    )

    if not jobs:
        raise SystemExit(f"No convertible episodes under {src}")

    run_t0 = time.perf_counter()
    results = _run_episode_jobs(jobs, episode_workers=episode_workers)
    timings = _finalize_metadata(dataset, jobs, results)
    wall_s = time.perf_counter() - run_t0

    for job in jobs:
        if profile:
            result = results[job.episode_index]
            tqdm.tqdm.write(
                f"{Path(job.ep_dir).name}: frames={result.episode_length} "
                f"numeric={result.timings['read_numeric']:.2f}s "
                f"decode={result.timings['decode']:.2f}s "
                f"table={result.timings['table']:.2f}s "
                f"stats={result.timings['stats']:.2f}s "
                f"encode={result.timings['encode']:.2f}s "
                f"worker_total={result.timings['total']:.2f}s"
            )

    video_files = list(dataset.root.rglob("*.mp4"))
    assert len(video_files) == len(jobs) * len(dataset.meta.video_keys)

    parquet_files = list(dataset.root.rglob("*.parquet"))
    assert len(parquet_files) == len(jobs)

    print(f"\nDataset written to {ds_path}")
    print(f"  Converted: {len(jobs)}")
    print(f"  Skipped:   {len(skipped)}")
    print("Timing totals:")
    for name in ("read_numeric", "decode", "table", "stats", "encode", "total"):
        print(f"  worker_{name:12s}: {timings[name]:.2f}s")
    print(f"  wall_clock    : {wall_s:.2f}s")
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
        "--episode-workers",
        type=int,
        default=128,
        help="Parallel episode worker processes. Use 1 for serial episode processing.",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=8,
        help="Parallel camera decode workers inside each episode worker",
    )
    parser.add_argument(
        "--video-encode-workers",
        type=int,
        default=8,
        help="Parallel direct MP4 encode workers inside each episode worker",
    )
    parser.add_argument("--resize-backend", choices=("none", "cv2", "pil"), default="none",
                        help="Deprecated compatibility flag; source resolution is always preserved.")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-gop", type=int, default=2)
    parser.add_argument("--robot-type", type=str, default=None, help="Override inferred LeRobot robot_type")
    parser.add_argument("--profile", action="store_true", help="Print per-episode timing breakdown")
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
        episode_workers=args.episode_workers,
    )


if __name__ == "__main__":
    main()
