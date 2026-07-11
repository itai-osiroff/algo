"""Headless bead-quality disagreement report for sweep metric rankings."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from moment_images.bead_quality_diagnostics import (
    BEAD_QUALITY_CSV,
    BeadQualityRecord,
    classify_bead_disagreement,
    find_bead_quality_csv,
    read_bead_quality_csv,
)
from moment_images.reliable_io import write_csv_atomic, write_json_atomic, write_text_atomic
from moment_images.sweep_metric_explorer import (
    TRIO_CANDIDATE_ROI_COVERAGE,
    TRIO_ORACLE_PROFILE_NAME,
    TRIO_VALID_FOV_FRACTION,
    _ensure_profile_results,
    load_sweep_metric_explorer_model,
)


BeadKey = Tuple[str, str, str, Optional[str]]


def _finite_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _load_bead_records(paths: Sequence[Path]) -> List[BeadQualityRecord]:
    records: List[BeadQualityRecord] = []
    searched: List[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            candidates = sorted(path.rglob(BEAD_QUALITY_CSV))
        else:
            found = find_bead_quality_csv(path)
            candidates = [found] if found is not None else []
        searched.append(str(path))
        for candidate in candidates:
            records.extend(read_bead_quality_csv(candidate))
    if not records:
        searched_text = ", ".join(searched) if searched else "<none>"
        raise ValueError(
            f"No {BEAD_QUALITY_CSV} files found near trio results: {searched_text}. "
            "Rerun trio_benchmark.py with the updated bead diagnostics export."
        )
    return records


def _bead_lookup(records: Iterable[BeadQualityRecord]) -> Dict[BeadKey, BeadQualityRecord]:
    lookup: Dict[BeadKey, BeadQualityRecord] = {}
    for record in records:
        if record.source_kind != "lf_candidate":
            continue
        lookup[(record.stage, record.candidate_id, record.dataset_id, record.generation_id)] = record
        lookup.setdefault((record.stage, record.candidate_id, record.dataset_id, None), record)
    return lookup


def _find_bead_record(
    lookup: Mapping[BeadKey, BeadQualityRecord],
    *,
    stage: str,
    candidate_id: str,
    dataset_id: str,
    generation_id: Optional[str],
) -> Optional[BeadQualityRecord]:
    return lookup.get((stage, candidate_id, dataset_id, generation_id)) or lookup.get(
        (stage, candidate_id, dataset_id, None)
    )


def _raw_record_metric_lookup(model) -> Dict[BeadKey, Mapping[str, float]]:
    lookup: Dict[BeadKey, Mapping[str, float]] = {}
    for record in model.raw_records:
        key = (record.stage, record.candidate_id, record.dataset_id, record.generation_id)
        lookup[key] = record.metrics
        lookup.setdefault((record.stage, record.candidate_id, record.dataset_id, None), record.metrics)
    return lookup


def _profile_ranking_for_dataset(profile_results: Sequence[object], dataset_id: str) -> List[object]:
    rows = [
        row
        for row in profile_results
        if dataset_id in row.dataset_scores and math.isfinite(float(row.dataset_scores[dataset_id]))
    ]
    return sorted(rows, key=lambda row: float(row.dataset_scores[dataset_id]), reverse=True)


def _rank_by_candidate(rows: Sequence[object]) -> Dict[str, int]:
    return {str(row.candidate_id): index for index, row in enumerate(rows, start=1)}


def _score_by_candidate(rows: Sequence[object], dataset_id: str) -> Dict[str, float]:
    return {str(row.candidate_id): float(row.dataset_scores[dataset_id]) for row in rows}


def _candidate_generation_id(rows: Sequence[object], candidate_id: str) -> Optional[str]:
    for row in rows:
        if str(row.candidate_id) == candidate_id:
            return row.generation_id
    return None


def _metric_value(
    lookup: Mapping[BeadKey, Mapping[str, float]],
    *,
    stage: str,
    candidate_id: str,
    dataset_id: str,
    generation_id: Optional[str],
    metric_name: str,
) -> Optional[float]:
    metrics = lookup.get((stage, candidate_id, dataset_id, generation_id)) or lookup.get(
        (stage, candidate_id, dataset_id, None)
    )
    if metrics is None:
        return None
    return _finite_float(metrics.get(metric_name))


def build_bead_disagreement_rows(
    model,
    *,
    bead_records: Sequence[BeadQualityRecord],
    oracle_profile: str,
    proxy_profile: str,
    top_k: int,
) -> List[Dict[str, object]]:
    """Build per-dataset proxy-vs-oracle bead disagreement rows."""

    bead_lookup = _bead_lookup(bead_records)
    raw_metric_lookup = _raw_record_metric_lookup(model)
    rows: List[Dict[str, object]] = []
    datasets = sorted({record.dataset_id for record in bead_records if record.source_kind == "lf_candidate"})
    stages = list(model.base_model.stage_order)
    for stage in stages:
        oracle_results = _ensure_profile_results(
            model,
            stage_name=stage,
            profile_name=oracle_profile,
            include_analysis=False,
        )
        proxy_results = _ensure_profile_results(
            model,
            stage_name=stage,
            profile_name=proxy_profile,
            include_analysis=False,
        )
        for dataset_id in datasets:
            oracle_ranked = _profile_ranking_for_dataset(oracle_results, dataset_id)
            proxy_ranked = _profile_ranking_for_dataset(proxy_results, dataset_id)
            if not oracle_ranked or not proxy_ranked:
                continue
            oracle_top = oracle_ranked[0]
            proxy_candidates = [
                row
                for row in proxy_ranked[: max(1, int(top_k))]
                if str(row.candidate_id) != str(oracle_top.candidate_id)
            ]
            if not proxy_candidates:
                continue
            proxy_top = proxy_candidates[0]
            oracle_ranks = _rank_by_candidate(oracle_ranked)
            proxy_ranks = _rank_by_candidate(proxy_ranked)
            oracle_scores = _score_by_candidate(oracle_ranked, dataset_id)
            proxy_scores = _score_by_candidate(proxy_ranked, dataset_id)
            oracle_generation = _candidate_generation_id(oracle_ranked, str(oracle_top.candidate_id))
            proxy_generation = _candidate_generation_id(proxy_ranked, str(proxy_top.candidate_id))
            oracle_bead = _find_bead_record(
                bead_lookup,
                stage=stage,
                candidate_id=str(oracle_top.candidate_id),
                dataset_id=dataset_id,
                generation_id=oracle_generation,
            )
            proxy_bead = _find_bead_record(
                bead_lookup,
                stage=stage,
                candidate_id=str(proxy_top.candidate_id),
                dataset_id=dataset_id,
                generation_id=proxy_generation,
            )
            if oracle_bead is None or proxy_bead is None:
                continue
            classification = classify_bead_disagreement(
                oracle_metrics=oracle_bead.metrics,
                proxy_metrics=proxy_bead.metrics,
                oracle_valid_fov_fraction=_metric_value(
                    raw_metric_lookup,
                    stage=stage,
                    candidate_id=str(oracle_top.candidate_id),
                    dataset_id=dataset_id,
                    generation_id=oracle_generation,
                    metric_name=TRIO_VALID_FOV_FRACTION,
                ),
                proxy_valid_fov_fraction=_metric_value(
                    raw_metric_lookup,
                    stage=stage,
                    candidate_id=str(proxy_top.candidate_id),
                    dataset_id=dataset_id,
                    generation_id=proxy_generation,
                    metric_name=TRIO_VALID_FOV_FRACTION,
                ),
                oracle_candidate_roi_coverage=_metric_value(
                    raw_metric_lookup,
                    stage=stage,
                    candidate_id=str(oracle_top.candidate_id),
                    dataset_id=dataset_id,
                    generation_id=oracle_generation,
                    metric_name=TRIO_CANDIDATE_ROI_COVERAGE,
                ),
                proxy_candidate_roi_coverage=_metric_value(
                    raw_metric_lookup,
                    stage=stage,
                    candidate_id=str(proxy_top.candidate_id),
                    dataset_id=dataset_id,
                    generation_id=proxy_generation,
                    metric_name=TRIO_CANDIDATE_ROI_COVERAGE,
                ),
            )
            rows.append(
                {
                    "stage": stage,
                    "dataset_id": dataset_id,
                    "oracle_profile": oracle_profile,
                    "proxy_profile": proxy_profile,
                    "oracle_candidate_id": oracle_top.candidate_id,
                    "proxy_candidate_id": proxy_top.candidate_id,
                    "oracle_rank": oracle_ranks.get(str(oracle_top.candidate_id), ""),
                    "proxy_rank": proxy_ranks.get(str(proxy_top.candidate_id), ""),
                    "proxy_candidate_oracle_rank": oracle_ranks.get(str(proxy_top.candidate_id), ""),
                    "oracle_candidate_proxy_rank": proxy_ranks.get(str(oracle_top.candidate_id), ""),
                    "oracle_score": oracle_scores.get(str(oracle_top.candidate_id), ""),
                    "proxy_score": proxy_scores.get(str(proxy_top.candidate_id), ""),
                    "classification": classification,
                    "oracle_validation_rank": oracle_bead.validation_rank or "",
                    "proxy_validation_rank": proxy_bead.validation_rank or "",
                    "oracle_compactness_score": oracle_bead.metrics.compactness_score,
                    "proxy_compactness_score": proxy_bead.metrics.compactness_score,
                    "oracle_width_excess_mean_px": oracle_bead.metrics.width_excess_mean_px,
                    "proxy_width_excess_mean_px": proxy_bead.metrics.width_excess_mean_px,
                    "oracle_support_area_ratio_mean": oracle_bead.metrics.support_area_ratio_mean,
                    "proxy_support_area_ratio_mean": proxy_bead.metrics.support_area_ratio_mean,
                    "oracle_ringing_penalty_mean": oracle_bead.metrics.ringing_penalty_mean,
                    "proxy_ringing_penalty_mean": proxy_bead.metrics.ringing_penalty_mean,
                    "oracle_bead_crop_path": oracle_bead.bead_crop_path or "",
                    "proxy_bead_crop_path": proxy_bead.bead_crop_path or "",
                }
            )
    rows.sort(key=lambda row: (str(row["stage"]), str(row["dataset_id"]), str(row["classification"])))
    return rows


def _fieldnames(rows: Sequence[Mapping[str, object]]) -> List[str]:
    preferred = [
        "stage",
        "dataset_id",
        "oracle_profile",
        "proxy_profile",
        "oracle_candidate_id",
        "proxy_candidate_id",
        "oracle_rank",
        "proxy_rank",
        "proxy_candidate_oracle_rank",
        "oracle_candidate_proxy_rank",
        "oracle_score",
        "proxy_score",
        "classification",
        "oracle_validation_rank",
        "proxy_validation_rank",
        "oracle_compactness_score",
        "proxy_compactness_score",
        "oracle_width_excess_mean_px",
        "proxy_width_excess_mean_px",
        "oracle_support_area_ratio_mean",
        "proxy_support_area_ratio_mean",
        "oracle_ringing_penalty_mean",
        "proxy_ringing_penalty_mean",
        "oracle_bead_crop_path",
        "proxy_bead_crop_path",
    ]
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return preferred + extras


def export_bead_diagnostic_report(
    model,
    *,
    bead_records: Sequence[BeadQualityRecord],
    oracle_profile: str,
    proxy_profile: str,
    output_dir: Path,
    top_k: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_bead_disagreement_rows(
        model,
        bead_records=bead_records,
        oracle_profile=oracle_profile,
        proxy_profile=proxy_profile,
        top_k=top_k,
    )
    csv_path = output_dir / "bead_disagreements.csv"
    write_csv_atomic(csv_path, _fieldnames(rows), rows, create_backup=True)
    summary = {
        "oracle_profile": oracle_profile,
        "proxy_profile": proxy_profile,
        "top_k": int(top_k),
        "bead_record_count": len(bead_records),
        "disagreement_count": len(rows),
        "classification_counts": {
            bucket: sum(1 for row in rows if row.get("classification") == bucket)
            for bucket in sorted({str(row.get("classification", "")) for row in rows})
        },
        "files": {
            "bead_disagreements_csv": csv_path.name,
            "index_html": "index.html",
        },
    }
    write_json_atomic(output_dir / "bead_diagnostic_summary.json", summary, indent=2, sort_keys=True, create_backup=True)
    _write_html(output_dir / "index.html", rows, summary)
    return output_dir


def _write_html(path: Path, rows: Sequence[Mapping[str, object]], summary: Mapping[str, object]) -> None:
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('stage', '')))}</td>"
        f"<td>{html.escape(str(row.get('dataset_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('oracle_candidate_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('proxy_candidate_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('classification', '')))}</td>"
        f"<td>{html.escape(str(row.get('proxy_compactness_score', '')))}</td>"
        "</tr>"
        for row in rows
    )
    if not table_rows:
        table_rows = "<tr><td colspan=\"6\">No proxy/oracle bead disagreements found.</td></tr>"
    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Bead Ranking Diagnostics</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
  </style>
</head>
<body>
  <h1>Bead Ranking Diagnostics</h1>
  <pre>{html.escape(json.dumps(summary, indent=2, sort_keys=True))}</pre>
  <table>
    <thead>
      <tr><th>Stage</th><th>Dataset</th><th>Oracle</th><th>Proxy</th><th>Classification</th><th>Proxy Compactness</th></tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""
    write_text_atomic(path, content, create_backup=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export bead-quality diagnostics for proxy-vs-oracle ranking disagreements.")
    parser.add_argument("sweep_dir", help="Sweep output directory to inspect.")
    parser.add_argument("--trio-results", action="append", required=True, help="trio_benchmark_results.csv or directory.")
    parser.add_argument("--proxy-profile", required=True, help="Profile to compare against the validation oracle profile.")
    parser.add_argument("--oracle-profile", default=TRIO_ORACLE_PROFILE_NAME)
    parser.add_argument("--metrics-config", default=None, help="Optional sweep_metric_explorer metrics config.")
    parser.add_argument("--output-dir", required=True, help="Directory for bead diagnostic report artifacts.")
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    trio_paths = [Path(item) for item in args.trio_results]
    model = load_sweep_metric_explorer_model(
        Path(args.sweep_dir),
        metrics_config_path=Path(args.metrics_config) if args.metrics_config else None,
        trio_results_paths=trio_paths,
    )
    bead_records = _load_bead_records(trio_paths)
    export_bead_diagnostic_report(
        model,
        bead_records=bead_records,
        oracle_profile=args.oracle_profile,
        proxy_profile=args.proxy_profile,
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
