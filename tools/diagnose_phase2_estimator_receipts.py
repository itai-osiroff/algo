"""Read-only diagnosis for the Phase 2 estimator receipts needed by Phase 3/4.

Run from the repository root in VS Code's terminal:
    python tools/diagnose_phase2_estimator_receipts.py /path/to/output/generations/<id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

# Running this file directly makes Python's initial import root ``tools/``.
# Include the repository root so the project modules are available in VS Code.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from moment_images.adaptive_power_blend_designer import AdaptiveCandidatePlan
from moment_images.adaptive_power_blend_protocol import (
    Phase02ExperimentContract,
    SpatialEstimatorEquivalenceReceipt,
    SpatialEstimatorResolutionReceipt,
)


DEFAULT_RUN_ROOT = Path(
    "/CPL_data_1/itaio/adaptive_power_blend_experiment_retry2/"
    "generations/gen_20260831_125133_run_0d32f135"
)


def describe_file(path: Path) -> dict[str, object]:
    """Return non-mutating file identity details, including missing files."""

    if not path.is_file():
        return {"exists": False, "size_bytes": None, "sha256": None}
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _payload_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def diagnose(run_root: Path) -> dict[str, object]:
    """Evaluate precisely the receipt invariant enforced by Phase 3 admission."""

    paths = {
        "contract": run_root / "adaptive_receipts" / "experiment_contract.json",
        "phase0_plan": run_root / "adaptive_phase0_candidate_manifest.json",
        "equivalence": run_root / "adaptive_receipts" / "spatial_estimator_equivalence.json",
        "resolution": run_root / "adaptive_receipts" / "spatial_estimator_resolution.json",
        "certificate": run_root / "adaptive_receipts" / "phase2_coverage_certificate.json",
    }
    report: dict[str, object] = {
        "run_root": str(run_root),
        "files": {name: describe_file(path) for name, path in paths.items()},
    }
    missing = [name for name, detail in report["files"].items() if not detail["exists"]]  # type: ignore[index]
    if missing:
        report["admission_ready"] = False
        report["failure"] = f"Missing required receipt(s): {', '.join(missing)}"
        return report

    try:
        contract = Phase02ExperimentContract.from_payload(_load_json(paths["contract"]))
        phase0_plan = AdaptiveCandidatePlan.from_payload(_load_json(paths["phase0_plan"]))
        equivalence = SpatialEstimatorEquivalenceReceipt.from_payload(
            _load_json(paths["equivalence"])
        )
        resolution = SpatialEstimatorResolutionReceipt.from_payload(
            _load_json(paths["resolution"])
        )
        certificate = _load_json(paths["certificate"])
        checks = {
            "equivalence_schema_is_v2": equivalence.schema_version == 2,
            "equivalence_passed": equivalence.passed,
            "equivalence_contract_matches": (
                equivalence.phase02_contract_sha256 == contract.experiment_sha256
            ),
            "equivalence_phase0_plan_matches": (
                equivalence.phase_zero_plan_signature == phase0_plan.plan_signature
            ),
            "resolution_embeds_same_equivalence": (
                resolution.equivalence_receipt.to_payload() == equivalence.to_payload()
            ),
            "resolution_equivalence_digest_matches": (
                resolution.equivalence_receipt_sha256
                == _payload_digest(equivalence.to_payload())
            ),
        }
        expected = certificate.get("artifact_dependencies", {}).get("estimator_resolution")
        actual = report["files"]["resolution"]["sha256"]  # type: ignore[index]
        checks["resolution_matches_certificate"] = expected == actual
        report["checks"] = checks
        failed = [name for name, passed in checks.items() if not passed]
        report["admission_ready"] = not failed
        report["failure"] = None if not failed else f"Failed checks: {', '.join(failed)}"
    except Exception as error:  # The parser detail is the diagnostic result.
        report["admission_ready"] = False
        report["failure"] = f"Receipt parsing failed: {type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", nargs="?", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    report = diagnose(args.run_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admission_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
