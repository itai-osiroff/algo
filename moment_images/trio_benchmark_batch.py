"""Manifest-driven batch runner for trio benchmark evaluation scenes."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from moment_images.reliable_io import write_csv_atomic, write_json_atomic, write_text_atomic
from moment_images.trio_benchmark import (
    DEFAULT_RESIDUAL_TRANSLATION_MAX_PX,
    DEFAULT_REGISTRATION_MIN_MATCHED_BEADS,
    DEFAULT_REGISTRATION_MIN_STRUCTURE_SCORE,
    DEFAULT_REGISTRATION_PROFILE,
    DEFAULT_REGISTRATION_COARSE_MAX_DIM,
    DEFAULT_REGISTRATION_REFINE_MAX_DIM,
    DEFAULT_REGISTRATION_PREFILTER_HYPOTHESES,
    DEFAULT_REGISTRATION_DENSE_HYPOTHESES,
    DEFAULT_REGISTRATION_REFINE_MAX_EVALS,
    DEFAULT_REGISTRATION_SUSPECT_RESIDUAL_PX,
    DEFAULT_REGISTRATION_ROTATION_SEARCH,
    DEFAULT_REGISTRATION_TOP_HYPOTHESES,
    DEFAULT_REGISTRATION_DIAGNOSTICS_TOP_N,
    DEFAULT_SCALE_SEARCH,
    DEFAULT_TOP_K,
    PROFILE_NAME,
    BenchmarkConfig,
    parse_rotation_search,
    parse_scale_search,
    run_benchmark,
)
from moment_images.sweep_visualization import GENERATION_SCOPE_ALL


def run_batch_from_manifest(manifest_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Run trio benchmarks for every scene in ``manifest_path``."""

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(manifest_path)
    manifest_base = manifest_path.parent
    defaults = _mapping(manifest.get("defaults", {}))
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("manifest must contain a non-empty 'scenes' list")

    scene_summaries: List[Dict[str, Any]] = []
    for index, raw_scene in enumerate(scenes, start=1):
        scene = _mapping(raw_scene)
        scene_id = str(scene.get("scene_id") or f"scene_{index:03d}")
        scene_dir = output_dir / _safe_scene_dir_name(scene_id)
        print(f"Starting scene {index}/{len(scenes)}: {scene_id}", file=sys.stderr, flush=True)
        try:
            benchmark_config = _scene_to_benchmark_config(
                scene,
                defaults,
                manifest_base=manifest_base,
                output_dir=scene_dir,
            )
            result = run_benchmark(benchmark_config)
            scene_summaries.append(_successful_scene_summary(scene_id, benchmark_config, scene_dir, result.summary))
            print(f"Finished scene {index}/{len(scenes)}: {scene_id}", file=sys.stderr, flush=True)
        except Exception as exc:
            scene_dir.mkdir(parents=True, exist_ok=True)
            scene_summaries.append(_failed_scene_summary(scene_id, scene, scene_dir, exc))
            print(f"Failed scene {index}/{len(scenes)}: {scene_id}: {type(exc).__name__}", file=sys.stderr, flush=True)

    summary = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "profile_name": PROFILE_NAME,
        "num_scenes": len(scene_summaries),
        "num_successful": sum(1 for scene in scene_summaries if scene["status"] == "ok"),
        "num_failed": sum(1 for scene in scene_summaries if scene["status"] != "ok"),
        "scenes": scene_summaries,
    }
    write_json_atomic(output_dir / "trio_batch_summary.json", summary, sort_keys=True, create_backup=False)
    _write_batch_csv(output_dir / "trio_batch_results.csv", scene_summaries)
    write_text_atomic(output_dir / "trio_batch_report.html", _render_batch_html(summary), create_backup=False)
    return summary


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"scenes": payload}
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object or a list of scene objects")
    return payload


def _scene_to_benchmark_config(
    scene: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    manifest_base: Path,
    output_dir: Path,
) -> BenchmarkConfig:
    merged = dict(defaults)
    merged.update(scene)
    sweep_dir = _required_path(merged, "sweep_dir", manifest_base)
    validation_image = _optional_path(merged.get("validation_image"), manifest_base)
    thorcam_image = _optional_path(merged.get("thorcam_image"), manifest_base)
    validation_workspace = _optional_path(merged.get("validation_workspace"), manifest_base)
    return BenchmarkConfig(
        sweep_dir=sweep_dir,
        dataset_id=str(_required_value(merged, "dataset_id")),
        validation_image=validation_image,
        thorcam_image=thorcam_image,
        validation_workspace=validation_workspace,
        output_dir=output_dir,
        stage=_optional_str(merged.get("stage")),
        generation_scope=str(merged.get("generation_scope", GENERATION_SCOPE_ALL)),
        anchor_row_id=str(merged.get("anchor_row_id", "auto")),
        scale_values=None if "scale_search" not in merged else parse_scale_search(str(merged.get("scale_search", DEFAULT_SCALE_SEARCH))),
        rotation_values=parse_rotation_search(str(merged.get("registration_rotation_search", DEFAULT_REGISTRATION_ROTATION_SEARCH))),
        top_k=int(merged.get("top_k", DEFAULT_TOP_K)),
        residual_translation_max_px=float(
            merged.get("residual_translation_max_px", DEFAULT_RESIDUAL_TRANSLATION_MAX_PX)
        ),
        registration_profile=str(merged.get("registration_profile", DEFAULT_REGISTRATION_PROFILE)),
        registration_suspect_residual_px=float(
            merged.get("registration_suspect_residual_px", DEFAULT_REGISTRATION_SUSPECT_RESIDUAL_PX)
        ),
        registration_min_matched_beads=int(
            merged.get("registration_min_matched_beads", DEFAULT_REGISTRATION_MIN_MATCHED_BEADS)
        ),
        registration_min_structure_score=float(
            merged.get("registration_min_structure_score", DEFAULT_REGISTRATION_MIN_STRUCTURE_SCORE)
        ),
        registration_top_hypotheses=int(
            merged.get("registration_top_hypotheses", DEFAULT_REGISTRATION_TOP_HYPOTHESES)
        ),
        registration_diagnostics_top_n=int(
            merged.get("registration_diagnostics_top_n", DEFAULT_REGISTRATION_DIAGNOSTICS_TOP_N)
        ),
        registration_coarse_max_dim=int(merged.get("registration_coarse_max_dim", DEFAULT_REGISTRATION_COARSE_MAX_DIM)),
        registration_refine_max_dim=int(merged.get("registration_refine_max_dim", DEFAULT_REGISTRATION_REFINE_MAX_DIM)),
        registration_prefilter_hypotheses=int(
            merged.get("registration_prefilter_hypotheses", DEFAULT_REGISTRATION_PREFILTER_HYPOTHESES)
        ),
        registration_dense_hypotheses=int(
            merged.get("registration_dense_hypotheses", DEFAULT_REGISTRATION_DENSE_HYPOTHESES)
        ),
        registration_refine_max_evals=int(
            merged.get("registration_refine_max_evals", DEFAULT_REGISTRATION_REFINE_MAX_EVALS)
        ),
        registration_progress=bool(merged.get("registration_progress", False)),
        enable_nonrigid_diagnostic=bool(merged.get("enable_nonrigid_diagnostic", False)),
    )


def _successful_scene_summary(
    scene_id: str,
    config: BenchmarkConfig,
    scene_dir: Path,
    benchmark_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    best = _mapping(benchmark_summary.get("best_candidate", {}))
    thorcam = _mapping(benchmark_summary.get("thorcam_baseline", {}))
    deltas = _mapping(benchmark_summary.get("best_lf_minus_thorcam", {}))
    best_metrics = _mapping(best.get("metrics", {}))
    thor_metrics = _mapping(thorcam.get("metrics", {}))
    best_registration = _mapping(best.get("registration", {}))
    return {
        "scene_id": scene_id,
        "status": "ok",
        "dataset_id": str(config.dataset_id),
        "sweep_dir": str(config.sweep_dir),
        "output_dir": str(scene_dir),
        "report_path": str(scene_dir / "trio_benchmark_report.html"),
        "summary_path": str(scene_dir / "trio_benchmark_summary.json"),
        "best_lf_row_id": best.get("row_id"),
        "best_lf_candidate_id": best.get("candidate_id"),
        "best_validation_hybrid_v1": best_metrics.get(PROFILE_NAME),
        "thorcam_validation_hybrid_v1": thor_metrics.get(PROFILE_NAME),
        "best_lf_minus_thorcam_validation_hybrid_v1": deltas.get(PROFILE_NAME),
        "registration_profile": config.registration_profile,
        "registration_confidence": best_registration.get("registration_confidence"),
        "registration_suspect": best_registration.get("registration_suspect"),
        "registration_constraint_level": best_registration.get("registration_constraint_level"),
        "registration_underconstrained": best_registration.get("registration_underconstrained"),
        "registration_initializer": best_registration.get("registration_initializer"),
        "registration_runtime_seconds": best_registration.get("registration_runtime_seconds"),
        "hypotheses_generated": best_registration.get("hypotheses_generated"),
        "hypotheses_prefiltered": best_registration.get("hypotheses_prefiltered"),
        "hypotheses_dense_scored": best_registration.get("hypotheses_dense_scored"),
        "hypotheses_refined": best_registration.get("hypotheses_refined"),
        "optimizer_evaluations": best_registration.get("optimizer_evaluations"),
        "warnings": list(benchmark_summary.get("warnings", [])),
        "error": None,
    }


def _failed_scene_summary(scene_id: str, scene: Mapping[str, Any], scene_dir: Path, exc: Exception) -> Dict[str, Any]:
    return {
        "scene_id": scene_id,
        "status": "error",
        "dataset_id": str(scene.get("dataset_id", "")),
        "sweep_dir": str(scene.get("sweep_dir", "")),
        "output_dir": str(scene_dir),
        "report_path": None,
        "summary_path": None,
        "best_lf_row_id": None,
        "best_lf_candidate_id": None,
        "best_validation_hybrid_v1": None,
        "thorcam_validation_hybrid_v1": None,
        "best_lf_minus_thorcam_validation_hybrid_v1": None,
        "registration_profile": None,
        "registration_confidence": None,
        "registration_suspect": None,
        "registration_constraint_level": None,
        "registration_underconstrained": None,
        "registration_initializer": None,
        "registration_runtime_seconds": None,
        "hypotheses_generated": None,
        "hypotheses_prefiltered": None,
        "hypotheses_dense_scored": None,
        "hypotheses_refined": None,
        "optimizer_evaluations": None,
        "warnings": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


def _write_batch_csv(path: Path, scene_summaries: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "scene_id",
        "status",
        "dataset_id",
        "sweep_dir",
        "output_dir",
        "report_path",
        "best_lf_row_id",
        "best_lf_candidate_id",
        "best_validation_hybrid_v1",
        "thorcam_validation_hybrid_v1",
        "best_lf_minus_thorcam_validation_hybrid_v1",
        "registration_profile",
        "registration_confidence",
        "registration_suspect",
        "registration_constraint_level",
        "registration_underconstrained",
        "registration_initializer",
        "registration_runtime_seconds",
        "hypotheses_generated",
        "hypotheses_prefiltered",
        "hypotheses_dense_scored",
        "hypotheses_refined",
        "optimizer_evaluations",
        "warnings",
        "error",
    ]
    rows = []
    for scene in scene_summaries:
        row = {name: scene.get(name, "") for name in fieldnames}
        row["warnings"] = "; ".join(str(warning) for warning in scene.get("warnings", []))
        rows.append(row)
    write_csv_atomic(path, fieldnames, rows, create_backup=False)


def _render_batch_html(summary: Mapping[str, Any]) -> str:
    rows = []
    for scene in summary.get("scenes", []):
        scene_map = _mapping(scene)
        report_path = scene_map.get("report_path")
        report_link = ""
        if report_path:
            scene_dir_name = Path(str(scene_map.get("output_dir", ""))).name
            report_href = f"{scene_dir_name}/{Path(str(report_path)).name}"
            report_link = f'<a href="{html.escape(report_href)}">report</a>'
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(scene_map.get('scene_id', '')))}</td>"
            f"<td>{html.escape(str(scene_map.get('status', '')))}</td>"
            f"<td>{html.escape(str(scene_map.get('best_lf_row_id', '') or ''))}</td>"
            f"<td>{_html_float(scene_map.get('best_validation_hybrid_v1'))}</td>"
            f"<td>{_html_float(scene_map.get('thorcam_validation_hybrid_v1'))}</td>"
            f"<td>{_html_float(scene_map.get('best_lf_minus_thorcam_validation_hybrid_v1'))}</td>"
            f"<td>{_html_float(scene_map.get('registration_confidence'))}</td>"
            f"<td>{html.escape(str(scene_map.get('registration_suspect') or ''))}</td>"
            f"<td>{html.escape(str(scene_map.get('registration_constraint_level') or ''))}</td>"
            f"<td>{report_link}</td>"
            f"<td>{html.escape(str(scene_map.get('error') or ''))}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Trio Benchmark Batch</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;text-align:left}"
        "th{background:#eee}</style></head><body>"
        "<h1>Trio Benchmark Batch</h1>"
        f"<p>Scenes: {int(summary.get('num_scenes', 0))}; "
        f"successful: {int(summary.get('num_successful', 0))}; "
        f"failed: {int(summary.get('num_failed', 0))}</p>"
        "<table><thead><tr><th>Scene</th><th>Status</th><th>Best LF Row</th>"
        "<th>Best LF Score</th><th>ThorCam Score</th><th>Delta</th><th>Reg Conf</th><th>Reg Suspect</th><th>Constraint</th>"
        "<th>Report</th><th>Error</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></body></html>"
    )


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _required_value(mapping: Mapping[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"scene is missing required field {key!r}")
    return value


def _required_path(mapping: Mapping[str, Any], key: str, base: Path) -> Path:
    return _resolve_path(_required_value(mapping, key), base)


def _optional_path(value: Any, base: Path) -> Optional[Path]:
    if value is None or str(value) == "":
        return None
    return _resolve_path(value, base)


def _resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _safe_scene_dir_name(scene_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", scene_id.strip())
    return safe or "scene"


def _html_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trio benchmark for a manifest of scenes.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_batch_from_manifest(args.manifest, args.output_dir)
    print(f"Wrote trio batch report to: {args.output_dir}")
    print(f"Scenes: {summary['num_scenes']}; successful: {summary['num_successful']}; failed: {summary['num_failed']}")


if __name__ == "__main__":
    main()
