"""Click-assisted annotation helper for manual trio registration calibration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from moment_images.reliable_io import write_json_atomic
from moment_images.trio_benchmark import (
    DEFAULT_REGISTRATION_CALIBRATION_GROUP,
    BenchmarkConfig,
    _collect_candidate_images,
    _load_reference_images,
    _select_anchor,
)
from moment_images.trio_benchmark_batch import _load_manifest, _mapping, _scene_to_benchmark_config
from moment_images.sweep_visualization import load_sweep_visualization_model


PointPair = Tuple[Tuple[float, float], Tuple[float, float]]


def build_annotation_record(
    *,
    scene: Mapping[str, Any],
    source_kind: str,
    anchor: Optional[Any],
    validation: np.ndarray,
    moving: np.ndarray,
    point_pairs: Sequence[PointPair],
) -> Dict[str, Any]:
    """Build one serializable annotation record from clicked point pairs.

    ``point_pairs`` are ordered as ``((reference_x, reference_y), (moving_x, moving_y))``.
    """

    source = str(source_kind).strip().lower()
    if source not in {"lf", "thorcam"}:
        raise ValueError("source_kind must be 'lf' or 'thorcam'")
    record: Dict[str, Any] = {
        "scene_id": str(scene.get("scene_id") or scene.get("dataset_id") or ""),
        "dataset_id": str(scene.get("dataset_id") or ""),
        "source_kind": source,
        "calibration_group": str(
            scene.get("registration_calibration_group")
            or scene.get("calibration_group")
            or DEFAULT_REGISTRATION_CALIBRATION_GROUP
        ),
        "reference_shape": [int(validation.shape[0]), int(validation.shape[1])],
        "moving_shape": [int(moving.shape[0]), int(moving.shape[1])],
        "points": [
            {
                "reference": [float(reference[0]), float(reference[1])],
                "moving": [float(moving_point[0]), float(moving_point[1])],
            }
            for reference, moving_point in point_pairs
        ],
    }
    if source == "lf" and anchor is not None:
        row_id = str(getattr(anchor, "row_id", ""))
        candidate_id = str(getattr(anchor, "candidate_id", ""))
        if row_id:
            record["anchor_row_id"] = row_id
        if candidate_id:
            record["anchor_candidate_id"] = candidate_id
    return record


def write_annotation_payload(output_path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    write_json_atomic(
        Path(output_path),
        {"version": 1, "annotations": list(records)},
        sort_keys=True,
        create_backup=False,
    )


def collect_point_pairs(validation: np.ndarray, moving: np.ndarray, *, title: str) -> List[PointPair]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(validation, cmap="gray")
    axes[0].set_title("validation/reference")
    axes[1].imshow(moving, cmap="gray")
    axes[1].set_title("moving")
    fig.suptitle(title)
    for axis in axes:
        axis.set_axis_off()
    pairs: List[PointPair] = []
    while True:
        print("Click a validation point, or press Enter/close the window to finish.")
        reference_points = plt.ginput(1, timeout=-1)
        if not reference_points:
            break
        print("Click the matching moving-image point.")
        moving_points = plt.ginput(1, timeout=-1)
        if not moving_points:
            break
        reference = (float(reference_points[0][0]), float(reference_points[0][1]))
        moving_point = (float(moving_points[0][0]), float(moving_points[0][1]))
        pairs.append((reference, moving_point))
        axes[0].plot(reference[0], reference[1], "ro", markersize=5)
        axes[1].plot(moving_point[0], moving_point[1], "go", markersize=5)
        fig.canvas.draw_idle()
    plt.close(fig)
    return pairs


def run_annotator_from_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    scene_ids: Optional[Sequence[str]] = None,
    max_scenes: Optional[int] = None,
    sources: Sequence[str] = ("lf", "thorcam"),
) -> Dict[str, Any]:
    manifest = _load_manifest(Path(manifest_path))
    defaults = _mapping(manifest.get("defaults", {}))
    raw_scenes = manifest.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise ValueError("manifest must contain a non-empty 'scenes' list")
    wanted = None if scene_ids is None else {str(scene_id) for scene_id in scene_ids}
    selected = []
    for index, raw_scene in enumerate(raw_scenes, start=1):
        scene = _mapping(raw_scene)
        scene_id = str(scene.get("scene_id") or f"scene_{index:03d}")
        if wanted is not None and scene_id not in wanted:
            continue
        selected.append(scene)
        if max_scenes is not None and len(selected) >= int(max_scenes):
            break

    records: List[Dict[str, Any]] = []
    manifest_base = Path(manifest_path).parent
    for scene in selected:
        merged_scene = dict(defaults)
        merged_scene.update(scene)
        config = _scene_to_benchmark_config(
            merged_scene,
            {},
            manifest_base=manifest_base,
            output_dir=Path(output_path).parent / "_annotation_preview",
        )
        scene_records = _annotate_scene(config, merged_scene, sources=sources)
        records.extend(scene_records)
    write_annotation_payload(output_path, records)
    return {"version": 1, "annotations": records}


def _annotate_scene(
    config: BenchmarkConfig,
    scene: Mapping[str, Any],
    *,
    sources: Sequence[str],
) -> List[Dict[str, Any]]:
    references = _load_reference_images(config)
    model = load_sweep_visualization_model(config.sweep_dir)
    stage = config.stage or str(model.primary_stage_name)
    candidate_images, _warnings = _collect_candidate_images(model, config, stage)
    anchor = _select_anchor(candidate_images, config.anchor_row_id) if candidate_images else None

    records: List[Dict[str, Any]] = []
    normalized_sources = _parse_sources(sources)
    if "lf" in normalized_sources:
        if anchor is None:
            raise ValueError(f"no LF candidates available for scene {scene.get('scene_id') or config.dataset_id}")
        pairs = collect_point_pairs(
            references.validation,
            anchor.image,
            title=f"{scene.get('scene_id') or config.dataset_id} LF",
        )
        records.append(
            build_annotation_record(
                scene=scene,
                source_kind="lf",
                anchor=anchor.row,
                validation=references.validation,
                moving=anchor.image,
                point_pairs=pairs,
            )
        )
    if "thorcam" in normalized_sources:
        pairs = collect_point_pairs(
            references.validation,
            references.thorcam,
            title=f"{scene.get('scene_id') or config.dataset_id} ThorCam",
        )
        records.append(
            build_annotation_record(
                scene=scene,
                source_kind="thorcam",
                anchor=None,
                validation=references.validation,
                moving=references.thorcam,
                point_pairs=pairs,
            )
        )
    return records


def _parse_sources(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        for part in str(value).split(","):
            source = part.strip().lower()
            if not source:
                continue
            if source not in {"lf", "thorcam"}:
                raise ValueError("sources must contain only 'lf' and/or 'thorcam'")
            if source not in result:
                result.append(source)
    return result or ["lf", "thorcam"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Click corresponding points for trio manual registration.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene-id", action="append", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--sources", default="lf,thorcam")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_annotator_from_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        scene_ids=args.scene_id,
        max_scenes=args.max_scenes,
        sources=_parse_sources([args.sources]),
    )
    print(f"Wrote {len(payload['annotations'])} annotation records to: {args.output}")


if __name__ == "__main__":
    main()
