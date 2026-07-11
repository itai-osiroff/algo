import json
import time
from pathlib import Path

import pytest

from moment_images.parameter_sweep import (
    SpecGuiDatasetState,
    SpecGuiMetricExperimentRowState,
    SpecGuiRelocationRuleState,
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
    state.run_mode = "extend"
    state.dry_run = True
    state.strategy_global = True
    state.pixel_similarity_window_sizes = "11,21"
    state.combined_global_thresholds = "0.1,0.2"
    state.refinement_method = "neighbor"
    state.refinement_extrapolation = "inside_only"

    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    assert "run_mode" not in payload
    assert "dry_run" not in payload
    assert roundtrip.run_mode == "normal"
    assert roundtrip.dry_run is False
    assert roundtrip.output_dir == state.output_dir
    assert roundtrip.strategy_pixel is True
    assert roundtrip.strategy_combined is True
    assert roundtrip.strategy_global is True
    assert roundtrip.stage_mode == "two_stage"
    assert roundtrip.pixel_similarity_window_sizes == "11,21"
    assert roundtrip.combined_global_thresholds == "0.1,0.2"
    assert payload["stages"]["refinement"] == {
        "method": "neighbor",
        "extrapolation": "inside_only",
    }
    assert roundtrip.refinement_method == "neighbor"
    assert roundtrip.refinement_extrapolation == "inside_only"
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
    assert experimental["formulations"][0]["family"] == "blurred_gradient_corr"
    assert experimental["formulations"][1]["family"] == "blurred_composite_corr"
    assert roundtrip.metric_experiments_enabled is True
    assert roundtrip.metric_experiments_shortlist_top_k == "3"
    assert len(roundtrip.metric_experiment_rows) == 2
    assert roundtrip.metric_experiment_rows[0].family == "blurred_gradient_corr"
    assert roundtrip.metric_experiment_rows[1].family == "blurred_composite_corr"


def test_state_to_spec_roundtrip_with_validation_oracle_sweep(tmp_path):
    state = _build_valid_state(tmp_path)
    state.validation_oracle_enabled = True
    state.validation_oracle_top_k = "4"
    state.validation_oracle_budget_per_dataset = "6"
    state.validation_oracle_defaults_json = json.dumps(
        {
            "registration_profile": "rigid_b",
            "residual_translation_max_px": 3.0,
        }
    )
    state.validation_oracle_scenes_json = json.dumps(
        [
            {
                "dataset_id": "dataset_a",
                "scene_id": "scene_a",
                "validation_image": str(tmp_path / "validation.tif"),
                "thorcam_image": str(tmp_path / "thorcam.tif"),
            }
        ]
    )

    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    oracle = payload["validation_oracle_sweep"]
    assert oracle["enabled"] is True
    assert oracle["top_k"] == 4
    assert oracle["budget_per_dataset"] == 6
    assert oracle["defaults"]["registration_profile"] == "rigid_b"
    assert oracle["scenes"][0]["dataset_id"] == "dataset_a"
    assert roundtrip.validation_oracle_enabled is True
    assert roundtrip.validation_oracle_top_k == "4"
    assert roundtrip.validation_oracle_budget_per_dataset == "6"
    assert json.loads(roundtrip.validation_oracle_scenes_json)[0]["scene_id"] == "scene_a"


def test_new_metric_experiment_gui_state_defaults_to_independent_intensity(tmp_path):
    state = _build_valid_state(tmp_path)
    state.metric_experiments_enabled = True
    state.metric_experiment_rows = [SpecGuiMetricExperimentRowState()]

    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    experimental = payload["experimental_metric_sweep"]
    assert experimental["candidate_source"] == "independent_search"
    assert experimental["search_stages"]["total_budget_per_dataset"] == int(
        state.metric_experiments_shortlist_top_k
    )
    assert experimental["formulations"] == [
        {
            "id": "intensity_default",
            "family": "intensity",
            "alphas": [1.0],
        }
    ]
    assert roundtrip.metric_experiments_candidate_source == "independent_search"
    assert roundtrip.metric_experiment_rows[0].formulation_id == "intensity_default"
    assert roundtrip.metric_experiment_rows[0].family == "intensity"
    assert roundtrip.metric_experiment_rows[0].alphas == "1.0"


def test_state_to_spec_roundtrip_with_scoring_profile(tmp_path):
    state = _build_valid_state(tmp_path)
    config_path = tmp_path / "metrics_config.json"
    config_path.write_text(
        json.dumps(
            {
                "profiles": [
                    {
                        "name": "guide_focus",
                        "label": "Guide Focus",
                        "metric_weights": {"vollath_f4_guide": 1.0},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    state.metric_profile_config_path = str(config_path)
    state.scoring_profile_name = "guide_focus"
    state.metric_scope = "global_only_metrics"

    issues = validate_state(state)
    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    assert issues == []
    assert payload["scoring"]["profile_name"] == "guide_focus"
    assert payload["scoring"]["profile_config_path"] == str(config_path)
    assert "metric_weights" not in payload["scoring"]
    assert roundtrip.scoring_profile_name == "guide_focus"
    assert roundtrip.metric_profile_config_path == str(config_path)
    assert roundtrip.metric_scope == "global_only_metrics"


def test_state_to_spec_roundtrip_with_relocation(tmp_path):
    state = _build_valid_state(tmp_path)
    state.adopt_relocated = True
    state.relocation_rules = [
        SpecGuiRelocationRuleState(
            old_root=str(tmp_path / "old"),
            new_root=str(tmp_path / "new"),
        )
    ]

    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    assert payload["relocation"]["adopt_relocated"] is True
    assert payload["relocation"]["path_mappings"][0]["old"] == str((tmp_path / "old").resolve())
    assert payload["relocation"]["path_mappings"][0]["new"] == str((tmp_path / "new").resolve())
    assert roundtrip.adopt_relocated is True
    assert roundtrip.relocation_rules[0].old_root == str((tmp_path / "old").resolve())
    assert roundtrip.relocation_rules[0].new_root == str((tmp_path / "new").resolve())


def test_validate_state_reports_invalid_relocation_mapping(tmp_path):
    state = _build_valid_state(tmp_path)
    state.relocation_rules = [
        SpecGuiRelocationRuleState(old_root=str(tmp_path / "old"), new_root="")
    ]

    issues = validate_state(state)

    assert any("Relocation row" in issue for issue in issues)


def test_state_to_spec_supports_dynamic_metric_weights(tmp_path):
    state = _build_valid_state(tmp_path)
    state.metric_weights = {"vollath_f4_guide": "0.7", "laplacian_variance": "0.3"}

    payload = state_to_spec_dict(state)
    roundtrip = spec_dict_to_state(payload)

    assert payload["scoring"]["metric_weights"] == {
        "vollath_f4_guide": 0.7,
        "laplacian_variance": 0.3,
    }
    assert roundtrip.metric_weights["vollath_f4_guide"] == "0.7"


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


def test_validate_state_reports_invalid_refinement_settings(tmp_path):
    state = _build_valid_state(tmp_path)
    state.refinement_method = "bad-method"
    state.refinement_extrapolation = "bad-edge-policy"

    issues = validate_state(state)

    assert any("Refinement method" in issue for issue in issues)
    assert any("Refinement extrapolation" in issue for issue in issues)


def test_validate_state_extension_requires_output_dir(tmp_path):
    state = _build_valid_state(tmp_path)
    state.run_mode = "extend"
    state.output_dir = ""

    issues = validate_state(state)

    assert any("Extend existing sweep requires an output directory" in issue for issue in issues)


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
    assert state.metric_experiment_rows[0].family == "blurred_gradient_corr"


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

    app._show_completion_dialog = lambda _output, dry_run=False: None  # type: ignore[method-assign]
    errors = []
    app.messagebox.showerror = lambda *args, **kwargs: errors.append(args)  # type: ignore[assignment]

    app.dataset_rows[0]["source_path_var"].set(str(tmp_path / "views.npy"))
    app.output_dir_entry.delete(0, app.tk.END)
    app.output_dir_entry.insert(0, str(tmp_path / "out"))

    calls = []

    def _run(spec, **kwargs):
        calls.append(kwargs)
        return tmp_path / "out"

    monkeypatch.setattr(ps, "run_sweep_experiment", _run)

    app._on_run()

    deadline = time.time() + 5.0
    while app._is_running and time.time() < deadline:
        root.update()
        time.sleep(0.05)

    assert app._is_running is False
    assert app.status_var.get() == "Completed"
    assert not errors
    assert calls == [{"extend_existing": False, "dry_run": False}]
    assert app.run_btn.cget("text") == "Run Sweep"
    root.destroy()


def test_gui_run_lifecycle_extension_passes_extend_flag(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    app._show_completion_dialog = lambda _output, dry_run=False: None  # type: ignore[method-assign]
    app.messagebox.showerror = lambda *args, **kwargs: None  # type: ignore[assignment]
    app.messagebox.showwarning = lambda *args, **kwargs: None  # type: ignore[assignment]
    app.messagebox.askyesno = lambda *args, **kwargs: True  # type: ignore[assignment]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "run_state.json").write_text("{}", encoding="utf-8")
    app.dataset_rows[0]["source_path_var"].set(str(tmp_path / "views.npy"))
    app.output_dir_entry.delete(0, app.tk.END)
    app.output_dir_entry.insert(0, str(out_dir))
    app.run_mode_var.set("extend")
    app.dry_run_var.set(False)
    app._update_run_button_text()

    calls = []

    def _run(spec, **kwargs):
        calls.append(kwargs)
        return out_dir

    monkeypatch.setattr(ps, "run_sweep_experiment", _run)

    app._on_run()
    deadline = time.time() + 5.0
    while app._is_running and time.time() < deadline:
        root.update()
        time.sleep(0.05)

    assert app._is_running is False
    assert calls == [{"extend_existing": True, "dry_run": False}]
    assert app.run_btn.cget("text") == "Extend Sweep"
    root.destroy()


def test_gui_run_lifecycle_extension_dry_run_passes_flags(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    shown_dialogs = []
    app._show_completion_dialog = lambda _output, dry_run=False: shown_dialogs.append(dry_run)  # type: ignore[method-assign]
    app.messagebox.showerror = lambda *args, **kwargs: None  # type: ignore[assignment]
    app.messagebox.showwarning = lambda *args, **kwargs: None  # type: ignore[assignment]
    app.messagebox.askyesno = lambda *args, **kwargs: pytest.fail("dry-run extension should not ask for destructive confirmation")  # type: ignore[assignment]

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "run_state.json").write_text("{}", encoding="utf-8")
    app.dataset_rows[0]["source_path_var"].set(str(tmp_path / "views.npy"))
    app.output_dir_entry.delete(0, app.tk.END)
    app.output_dir_entry.insert(0, str(out_dir))
    app.run_mode_var.set("extend")
    app.dry_run_var.set(True)
    app._update_run_button_text()

    calls = []

    def _run(spec, **kwargs):
        calls.append(kwargs)
        generation_dir = out_dir / "generations" / "gen_test"
        generation_dir.mkdir(parents=True)
        (generation_dir / "dry_run.json").write_text("{}", encoding="utf-8")
        return out_dir

    monkeypatch.setattr(ps, "run_sweep_experiment", _run)

    app._on_run()
    deadline = time.time() + 5.0
    while app._is_running and time.time() < deadline:
        root.update()
        time.sleep(0.05)

    assert app._is_running is False
    assert calls == [{"extend_existing": True, "dry_run": True}]
    assert shown_dialogs == [True]
    assert app.run_btn.cget("text") == "Preview Run"
    root.destroy()


def test_gui_update_planned_runs_uses_preview_builder(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    app.dataset_rows[0]["source_path_var"].set(str(tmp_path / "views.npy"))
    app.output_dir_entry.delete(0, app.tk.END)
    app.output_dir_entry.insert(0, str(tmp_path / "out"))
    app.dry_run_var.set(False)

    calls = []

    def _preview(spec, **kwargs):
        calls.append((spec, kwargs))
        return {
            "stages": {
                "stage1": {
                    "planned_run_count": 2,
                    "pending_run_count": 2,
                    "skipped_valid_count": 0,
                    "exact": True,
                }
            },
            "metric_experiments": {
                "enabled": True,
                "candidate_source": "baseline_shortlist",
                "status": "estimated_requires_baseline_rankings",
                "num_planned_runs": 3,
                "pending_run_count": 3,
                "skipped_valid_count": 0,
                "exact": False,
                "estimate_reason": "baseline_rankings_unavailable",
            },
            "totals": {
                "planned_run_count": 5,
                "pending_run_count": 5,
                "skipped_valid_count": 0,
                "exact": False,
            },
        }

    monkeypatch.setattr(ps, "build_sweep_run_preview", _preview)
    monkeypatch.setattr(
        ps,
        "run_sweep_experiment",
        lambda *args, **kwargs: pytest.fail("planned run update must not start a sweep"),
    )

    app._on_update_planned_runs()
    deadline = time.time() + 5.0
    while app._is_previewing and time.time() < deadline:
        root.update()
        time.sleep(0.05)

    assert len(calls) == 1
    assert calls[0][1]["extend_existing"] is False
    assert calls[0][1]["allow_dataset_generation"] is False
    assert calls[0][1]["reuse_existing_artifacts"] is False
    assert "Pending: 5" in app.planned_runs_var.get()
    assert "Metric: estimated 3 pending" in app.planned_runs_detail_var.get()
    root.destroy()


def test_gui_update_planned_runs_reports_validation_errors(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    app.output_dir_entry.delete(0, app.tk.END)
    app.output_dir_entry.insert(0, str(tmp_path / "out"))
    app.dataset_rows[0]["source_path_var"].set("")
    errors = []
    app.messagebox.showerror = lambda *args, **kwargs: errors.append(args)  # type: ignore[assignment]
    monkeypatch.setattr(
        ps,
        "build_sweep_run_preview",
        lambda *args, **kwargs: pytest.fail("invalid state must not call preview builder"),
    )

    app._on_update_planned_runs()

    assert errors
    assert app._is_previewing is False
    assert "Plan unavailable" in app.planned_runs_var.get()
    root.destroy()


def test_gui_run_lifecycle_passes_relocation_options(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    app._show_completion_dialog = lambda _output, dry_run=False: None  # type: ignore[method-assign]
    app.messagebox.showerror = lambda *args, **kwargs: None  # type: ignore[assignment]

    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    app.dataset_rows[0]["source_path_var"].set(str(old_root / "views.npy"))
    app.output_dir_entry.delete(0, app.tk.END)
    app.output_dir_entry.insert(0, str(old_root / "out"))
    app._add_relocation_row(
        SpecGuiRelocationRuleState(old_root=str(old_root), new_root=str(new_root))
    )
    app.adopt_relocated_var.set(True)

    calls = []

    def _run(spec, **kwargs):
        calls.append((spec, kwargs))
        return new_root / "out"

    monkeypatch.setattr(ps, "run_sweep_experiment", _run)

    app._on_run()
    deadline = time.time() + 5.0
    while app._is_running and time.time() < deadline:
        root.update()
        time.sleep(0.05)

    assert app._is_running is False
    spec, kwargs = calls[0]
    assert spec.output_dir == str((new_root / "out").resolve())
    assert spec.datasets[0].views_path == str((new_root / "views.npy").resolve())
    assert kwargs["adopt_relocated"] is True
    assert kwargs["relocation_rules"][0].old_root == str(old_root.resolve())
    assert kwargs["relocation_rules"][0].new_root == str(new_root.resolve())
    assert kwargs["relocation_report"]["applied"]
    root.destroy()


def test_gui_extension_warns_when_output_dir_has_no_run_state(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    app._show_completion_dialog = lambda _output, dry_run=False: None  # type: ignore[method-assign]
    warnings = []
    app.messagebox.showerror = lambda *args, **kwargs: None  # type: ignore[assignment]
    app.messagebox.showwarning = lambda *args, **kwargs: warnings.append(args)  # type: ignore[assignment]
    app.messagebox.askyesno = lambda *args, **kwargs: True  # type: ignore[assignment]

    out_dir = tmp_path / "new_out"
    out_dir.mkdir()
    app.dataset_rows[0]["source_path_var"].set(str(tmp_path / "views.npy"))
    app.output_dir_entry.delete(0, app.tk.END)
    app.output_dir_entry.insert(0, str(out_dir))
    app.run_mode_var.set("extend")
    app.dry_run_var.set(False)

    monkeypatch.setattr(ps, "run_sweep_experiment", lambda spec, **kwargs: out_dir)

    app._on_run()
    deadline = time.time() + 5.0
    while app._is_running and time.time() < deadline:
        root.update()
        time.sleep(0.05)

    assert any("run_state.json" in str(args) for args in warnings)
    root.destroy()


def test_gui_run_lifecycle_failure(monkeypatch, tmp_path):
    from moment_images import parameter_sweep as ps

    root = _make_root_or_skip()
    app = ps.SpecGuiApp(root)

    errors = []
    app.messagebox.showerror = lambda *args, **kwargs: errors.append(args)  # type: ignore[assignment]
    app.dataset_rows[0]["source_path_var"].set(str(tmp_path / "views.npy"))

    def _raise(_spec, **_kwargs):
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
