import json
import time
from pathlib import Path

import pytest

from moment_images.parameter_sweep import (
    SpecGuiDatasetState,
    SpecGuiMetricExperimentRowState,
    SpecGuiState,
    _load_spec_payload,
    _save_spec_payload,
    spec_dict_to_state,
    state_to_spec_dict,
    validate_state,
)


def _build_valid_state(tmp_path: Path) -> SpecGuiState:
    state = SpecGuiState()
    state.output_dir = str(tmp_path / "out")
    state.datasets = [
        SpecGuiDatasetState(
            dataset_id="dataset_a",
            source_type="views",
            source_path=str(tmp_path / "views.npy"),
            normalize_to_8bit=False,
        )
    ]
    return state


def test_state_to_spec_and_back_roundtrip(tmp_path):
    state = _build_valid_state(tmp_path)
    state.strategy_global = True
    state.pixel_similarity_window_sizes = "11,21"
    state.combined_global_thresholds = "0.1,0.2"

    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    assert roundtrip.output_dir == state.output_dir
    assert roundtrip.strategy_pixel is True
    assert roundtrip.strategy_combined is True
    assert roundtrip.strategy_global is True
    assert roundtrip.stage_mode == "two_stage"
    assert roundtrip.pixel_similarity_window_sizes == "11,21"
    assert roundtrip.combined_global_thresholds == "0.1,0.2"
    assert roundtrip.datasets[0].dataset_id == "dataset_a"
    assert roundtrip.metric_experiments_enabled is False


def test_state_to_spec_roundtrip_supports_single_stage_mode(tmp_path):
    state = _build_valid_state(tmp_path)
    state.stage_mode = "single_stage"
    state.stage1_fraction = "not-a-number"

    issues = validate_state(state)
    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    assert not any("Stage1 fraction" in issue for issue in issues)
    assert payload["stages"]["mode"] == "single_stage"
    assert payload["stages"]["stage1_fraction"] == 0.6
    assert roundtrip.stage_mode == "single_stage"


def test_state_to_spec_roundtrip_with_metric_experiments(tmp_path):
    state = _build_valid_state(tmp_path)
    state.metric_experiments_enabled = True
    state.metric_experiments_shortlist_top_k = "3"
    state.metric_experiment_rows = [
        SpecGuiMetricExperimentRowState(
            formulation_id="grad_alpha",
            family="ncc_gradient",
            alphas="0.0,0.5,1.0",
        ),
        SpecGuiMetricExperimentRowState(
            formulation_id="comp_alpha",
            family="composite",
            alphas="0.25,0.75",
            gradient_weight="0.7",
            laplacian_weight="0.3",
        ),
    ]

    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    experimental = payload["experimental_metric_sweep"]
    assert experimental["enabled"] is True
    assert experimental["shortlist_top_k"] == 3
    assert len(experimental["formulations"]) == 2
    assert roundtrip.metric_experiments_enabled is True
    assert roundtrip.metric_experiments_shortlist_top_k == "3"
    assert len(roundtrip.metric_experiment_rows) == 2
    assert roundtrip.metric_experiment_rows[1].family == "composite"


def test_validate_state_reports_duplicate_and_missing_fields(tmp_path):
    state = _build_valid_state(tmp_path)
    state.datasets = [
        SpecGuiDatasetState(
            dataset_id="dup",
            source_type="source_image",
            source_path="",
            center_x="",
            center_y="10",
            view_radius="",
        ),
        SpecGuiDatasetState(
            dataset_id="dup",
            source_type="views",
            source_path="x.npy",
        ),
    ]

    issues = validate_state(state)

    assert any("Dataset IDs must be unique" in issue for issue in issues)
    assert any("source path is required" in issue for issue in issues)
    assert any("center_x is required" in issue for issue in issues)
    assert any("view_radius is required" in issue for issue in issues)


def test_validate_state_experimental_metrics_rules(tmp_path):
    state = _build_valid_state(tmp_path)
    state.strategy_pixel = False
    state.strategy_combined = False
    state.strategy_global = True
    state.metric_experiments_enabled = True
    state.metric_experiments_shortlist_top_k = "0"
    state.metric_experiment_rows = [
        SpecGuiMetricExperimentRowState(
            formulation_id="",
            family="composite",
            alphas="1.2",
            gradient_weight="-1",
            laplacian_weight="0",
        )
    ]

    issues = validate_state(state)

    assert any("pixel/combined strategy" in issue for issue in issues)
    assert any("shortlist top-k" in issue for issue in issues)
    assert any("id is required" in issue for issue in issues)
    assert any("alphas" in issue and "[0, 1]" in issue for issue in issues)
    assert any("non-negative" in issue for issue in issues)


def test_state_to_spec_maps_advanced_search_controls(tmp_path):
    state = _build_valid_state(tmp_path)
    state.pixel_sub_pix_factors = "1,2,3"
    state.pixel_sigma_pyramid = "true,false"
    state.pixel_zero_fill_roll = "true"
    state.combined_sub_pix_factors = "1,2"
    state.combined_sigma_pyramid = "false,true"
    state.combined_zero_fill_roll = "true,false"
    state.combined_use_global_avg_reference = "true"

    payload = state_to_spec_dict(state)
    pixel = payload["search_spaces"]["pixel"]
    combined = payload["search_spaces"]["combined"]

    assert pixel["sub_pix_factors"] == [1, 2, 3]
    assert pixel["pixel_sigma_pyramid"] == [True, False]
    assert pixel["zero_fill_roll"] == [True]
    assert combined["sub_pix_factors"] == [1, 2]
    assert combined["pixel_sigma_pyramid"] == [False, True]
    assert combined["zero_fill_roll"] == [True, False]
    assert combined["combined_use_global_avg_reference"] == [True]


def test_save_and_load_spec_payload(tmp_path):
    payload = {
        "output_dir": str(tmp_path / "out"),
        "strategies": ["pixel", "combined"],
        "datasets": [{"id": "dataset_a", "views": "x.npy"}],
        "search_spaces": {},
    }
    path = tmp_path / "spec.json"
    _save_spec_payload(path, payload)
    loaded = _load_spec_payload(path)
    assert loaded == payload


def test_spec_dict_to_state_supports_legacy_metric_formulations(tmp_path):
    payload = {
        "output_dir": str(tmp_path / "out"),
        "strategies": ["pixel"],
        "datasets": [{"id": "dataset_a", "views": "x.npy"}],
        "search_spaces": {
            "pixel": {
                "similarity_window_sizes": [15],
                "max_displacements": [5],
                "gauss_sigmas": [3.0],
                "disp_gauss_sigmas": [0.0],
                "pixel_thresholds": [0.05],
                "sub_pix_factors": [1],
                "pixel_sigma_pyramid": [False],
                "zero_fill_roll": [True],
            }
        },
        "metric_formulations": [
            {"id": "legacy_grad", "family": "ncc_gradient", "alphas": [0.5]}
        ],
    }

    state = spec_dict_to_state(payload)

    assert state.metric_experiments_enabled is True
    assert len(state.metric_experiment_rows) == 1
    assert state.metric_experiment_rows[0].formulation_id == "legacy_grad"


def _can_create_tk():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


def _make_root_or_skip():
    if not _can_create_tk():
        pytest.skip("Tkinter UI unavailable in this environment")
    import tkinter as tk
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except Exception:
        pytest.skip("Tkinter root unavailable in this environment")


def test_gui_run_lifecycle_success(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    app._show_completion_dialog = lambda _output: None  # type: ignore[method-assign]
    errors = []
    app.messagebox.showerror = lambda *args, **kwargs: errors.append(args)  # type: ignore[assignment]

    app.dataset_rows[0]["source_path_var"].set(str(tmp_path / "views.npy"))
    app.output_dir_entry.delete(0, app.tk.END)
    app.output_dir_entry.insert(0, str(tmp_path / "out"))

    monkeypatch.setattr(ps, "run_sweep_experiment", lambda spec: tmp_path / "out")

    app._on_run()

    deadline = time.time() + 5.0
    while app._is_running and time.time() < deadline:
        root.update()
        time.sleep(0.05)

    assert app._is_running is False
    assert app.status_var.get() == "Completed"
    assert not errors
    root.destroy()


def test_gui_run_lifecycle_failure(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    errors = []
    app.messagebox.showerror = lambda *args, **kwargs: errors.append(args)  # type: ignore[assignment]
    app.dataset_rows[0]["source_path_var"].set(str(tmp_path / "views.npy"))

    def _raise(_spec):
        raise RuntimeError("boom")

    monkeypatch.setattr(ps, "run_sweep_experiment", _raise)

    app._on_run()
    deadline = time.time() + 5.0
    while app._is_running and time.time() < deadline:
        root.update()
        time.sleep(0.05)

    assert app._is_running is False
    assert app.status_var.get() == "Failed"
    assert errors
    root.destroy()
