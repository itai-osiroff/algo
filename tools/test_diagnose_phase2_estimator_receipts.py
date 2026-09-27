from pathlib import Path
import subprocess
import sys

from tools.diagnose_phase2_estimator_receipts import describe_file


def test_describe_file_reports_a_missing_receipt(tmp_path: Path) -> None:
    report = describe_file(tmp_path / "missing.json")

    assert report == {"exists": False, "size_bytes": None, "sha256": None}


def test_script_runs_from_the_repository_root_for_a_missing_receipt(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/diagnose_phase2_estimator_receipts.py",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert '"admission_ready": false' in result.stdout
    assert "Missing required receipt(s)" in result.stdout
