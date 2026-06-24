import json

import cv2 as cv
import numpy as np
import pytest

import moment_images.metric_profiles as profiles
from moment_images.metric_profiles import (
    RankingProfileSpec,
    build_metric_registry,
    compute_extended_image_metrics,
    resolve_profile_weights_for_sweep,
    validate_profile_for_sweep,
)
from moment_images.sweep_metric_explorer import compute_metrics_from_workspace


def test_extended_metrics_match_workspace_explorer(tmp_path):
    image = np.arange(64, dtype=np.float64).reshape(8, 8)
    workspace_path = tmp_path / "workspace.npz"
    np.savez(workspace_path, average_rectified_image=image)

    direct = compute_extended_image_metrics(image)
    explorer_metrics, warnings = compute_metrics_from_workspace(workspace_path)

    assert warnings == []
    for name in [
        "laplacian_variance",
        "ndct_shannon_avg",
        "ndct_bayes_guide",
        "gini_guide",
        "hoyer_avg",
        "vollath_f4_guide",
        "tamura_gradient_avg",
        "tamura_guide",
    ]:
        assert explorer_metrics[name] == pytest.approx(direct[name])


def test_tamura_metrics_and_profiles_are_sweep_compatible():
    sharp = np.zeros((32, 32), dtype=np.float64)
    sharp[10:22, 14:18] = 255.0
    blurred = cv.GaussianBlur(sharp.astype(np.float32), (0, 0), 2.5)

    assert profiles.tamura_coefficient(np.zeros((8, 8), dtype=np.float64)) == 0.0
    assert profiles.tamura_coefficient(np.full((8, 8), 3.0, dtype=np.float64)) == 0.0

    sharp_metrics = compute_extended_image_metrics(sharp)
    blurred_metrics = compute_extended_image_metrics(blurred)

    assert sharp_metrics["tamura_gradient_avg"] > blurred_metrics["tamura_gradient_avg"]
    assert sharp_metrics["tamura_guide"] > blurred_metrics["tamura_guide"]

    registry = build_metric_registry()
    assert registry["tamura_gradient_avg"].category == "tamura"
    assert registry["tamura_guide"].category == "tamura"

    weights, profile, warnings = resolve_profile_weights_for_sweep("tamura_tog")
    assert warnings == []
    assert profile.name == "tamura_tog"
    assert weights == {"tamura_gradient_avg": 1.0}

    weights, profile, warnings = resolve_profile_weights_for_sweep("tamura_focus_pair")
    assert warnings == []
    assert profile.name == "tamura_focus_pair"
    assert weights == {"tamura_gradient_avg": 0.5, "tamura_guide": 0.5}


def test_resolve_profile_weights_supports_custom_config_and_negative_opt_in(tmp_path):
    config_path = tmp_path / "metrics_config.json"
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "signed_focus",
                        "label": "Signed Focus",
                        "allow_negative_weights": True,
                        "metric_weights": {
                            "vollath_f4_guide": 0.8,
                            "average_intensity": -0.2,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    weights, profile, warnings = resolve_profile_weights_for_sweep(
        "signed_focus",
        config_path=config_path,
    )

    assert warnings == []
    assert profile.name == "signed_focus"
    assert weights == {"vollath_f4_guide": 0.8, "average_intensity": -0.2}


def test_saved_only_profile_is_rejected_for_sweep_ranking():
    registry = build_metric_registry()
    profile = RankingProfileSpec(
        name="saved_original",
        label="Saved Original",
        metric_weights={"saved_score": 1.0},
    )

    issues = validate_profile_for_sweep(profile, registry)

    assert any("saved-only" in issue for issue in issues)
    assert any("no sweep-compatible" in issue for issue in issues)
