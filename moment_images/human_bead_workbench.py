"""Tkinter workbench for faster human bead annotation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from moment_images.human_bead_oracle import (
    HumanBead,
    HumanBeadDatasetAnnotations,
    _load_image,
    _robust_unit,
    load_human_bead_annotations,
    save_human_bead_preview_sheet,
    write_human_bead_annotations,
)


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MIN_BEAD_RADIUS_PX = 0.5
GENERIC_PREVIEW_STEMS = frozenset(
    {
        "avg_rectified",
        "average_rectified_image",
        "rectified_image",
        "rectified",
        "preview",
    }
)
_GENERATED_DATASET_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_GENERATED_DATASET_ID_UNDERSCORES = re.compile(r"_+")


def _sanitize_generated_dataset_id(value: str) -> str:
    text = _GENERATED_DATASET_ID_UNSAFE.sub("_", str(value).strip())
    text = _GENERATED_DATASET_ID_UNDERSCORES.sub("_", text).strip("._-")
    return text or "dataset"


def _normalized_path_key(path: Path) -> str:
    item = Path(path).expanduser()
    try:
        item = item.resolve(strict=False)
    except OSError:
        pass
    return str(item).replace("\\", "/").casefold()


def _sweep_run_context(path: Path) -> Optional[Tuple[str, str, str]]:
    parts = list(Path(path).parts)
    lowered = [part.lower() for part in parts]
    for index in reversed(range(max(0, len(parts) - 4))):
        if lowered[index] == "runs" and index + 4 < len(parts):
            return parts[index + 1], parts[index + 2], parts[index + 3]
    return None


def _dedupe_candidate_ids(candidates: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for candidate in candidates:
        dataset_id = _sanitize_generated_dataset_id(candidate)
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        result.append(dataset_id)
    return result


def _base_generated_dataset_id(path: Path) -> str:
    item = Path(path)
    stem = item.stem
    if stem.lower() not in GENERIC_PREVIEW_STEMS:
        return _sanitize_generated_dataset_id(stem)
    sweep_context = _sweep_run_context(item)
    if sweep_context is not None:
        _stage, _candidate, dataset = sweep_context
        return _sanitize_generated_dataset_id(dataset)
    parent_name = item.parent.name
    return _sanitize_generated_dataset_id(parent_name or stem)


def _generated_dataset_id_candidates(path: Path, base_id: str) -> List[str]:
    item = Path(path)
    candidates: List[str] = []
    sweep_context = _sweep_run_context(item)
    if sweep_context is not None:
        stage, candidate, _dataset = sweep_context
        candidates.extend(
            [
                f"{base_id}_{candidate}",
                f"{base_id}_{stage}_{candidate}",
            ]
        )
    parent_name = item.parent.name
    grandparent_name = item.parent.parent.name
    if parent_name:
        candidates.append(f"{base_id}_{parent_name}")
    if grandparent_name and parent_name:
        candidates.append(f"{base_id}_{grandparent_name}_{parent_name}")
    return _dedupe_candidate_ids(candidates)


def _generated_dataset_id_for_path(path: Path, existing_dataset_images: Mapping[str, Path]) -> str:
    new_path_key = _normalized_path_key(Path(path))
    for dataset_id, existing_path in existing_dataset_images.items():
        if _normalized_path_key(Path(existing_path)) == new_path_key:
            return dataset_id

    base_id = _base_generated_dataset_id(Path(path))
    if base_id not in existing_dataset_images:
        return base_id

    for candidate_id in _generated_dataset_id_candidates(Path(path), base_id):
        if candidate_id not in existing_dataset_images:
            return candidate_id

    suffix = 2
    while f"{base_id}_{suffix}" in existing_dataset_images:
        suffix += 1
    return f"{base_id}_{suffix}"


def collect_dataset_images_from_folder(folder: Path) -> Dict[str, Path]:
    """Return supported images in a folder keyed by stable dataset id."""

    root = Path(folder)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"dataset folder does not exist: {root}")
    result: Dict[str, Path] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            result[path.stem] = path
    return result


def parse_dataset_image_values(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset-image must use dataset_id=path")
        dataset_id, path_text = value.split("=", 1)
        dataset_id = dataset_id.strip()
        if not dataset_id:
            raise ValueError("--dataset-image dataset_id cannot be empty")
        result[dataset_id] = Path(path_text)
    return result


def collect_dataset_images(
    dataset_image_values: Sequence[str],
    dataset_folders: Sequence[Path],
) -> Dict[str, Path]:
    """Merge explicit dataset images with folder imports.

    Explicit ``--dataset-image`` entries keep their insertion order and win over
    folder-discovered files with the same dataset id.
    """

    result = parse_dataset_image_values(dataset_image_values)
    for folder in dataset_folders:
        for dataset_id, path in collect_dataset_images_from_folder(Path(folder)).items():
            if dataset_id not in result:
                result[dataset_id] = path
    return result


def _clone_annotation(annotation: HumanBeadDatasetAnnotations) -> HumanBeadDatasetAnnotations:
    return HumanBeadDatasetAnnotations(
        dataset_id=annotation.dataset_id,
        image_path=annotation.image_path,
        image_shape=tuple(annotation.image_shape),
        default_radius_px=float(annotation.default_radius_px),
        beads=list(annotation.beads),
    )


def _image_shape(path: Path) -> Tuple[int, int]:
    image = _load_image(Path(path))
    return int(image.shape[0]), int(image.shape[1])


@dataclass(frozen=True)
class _HistoryEntry:
    dataset_id: str
    annotation: HumanBeadDatasetAnnotations
    selected_index: Optional[int]


class HumanBeadWorkbenchState:
    """Pure annotation-editing state used by the GUI and unit tests."""

    def __init__(
        self,
        dataset_images: Mapping[str, Path],
        output_path: Path,
        *,
        existing: Optional[Mapping[str, HumanBeadDatasetAnnotations]] = None,
        image_shapes: Optional[Mapping[str, Tuple[int, int]]] = None,
        default_radius_px: float = 4.0,
    ) -> None:
        self.dataset_ids = list(dataset_images)
        self.dataset_images = {dataset_id: Path(path) for dataset_id, path in dataset_images.items()}
        self.output_path = Path(output_path)
        self.annotations: Dict[str, HumanBeadDatasetAnnotations] = {
            dataset_id: _clone_annotation(annotation)
            for dataset_id, annotation in (existing or {}).items()
        }
        self.image_shapes = dict(image_shapes or {})
        self.default_radius_px = float(default_radius_px)
        self.current_index = 0
        self.selected_index: Optional[int] = None
        self.dirty = False
        self.dirty_dataset_ids: set[str] = set()
        self._undo_stack: List[_HistoryEntry] = []
        self._redo_stack: List[_HistoryEntry] = []

        if self.default_radius_px <= 0.0:
            raise ValueError("default_radius_px must be positive.")

        for dataset_id in self.dataset_ids:
            if dataset_id in self.annotations:
                continue
            shape = self.image_shapes.get(dataset_id)
            if shape is None:
                shape = _image_shape(self.dataset_images[dataset_id])
                self.image_shapes[dataset_id] = shape
            self.annotations[dataset_id] = HumanBeadDatasetAnnotations(
                dataset_id=dataset_id,
                image_path=str(self.dataset_images[dataset_id]),
                image_shape=(int(shape[0]), int(shape[1])),
                default_radius_px=self.default_radius_px,
                beads=[],
            )

    @property
    def current_dataset_id(self) -> str:
        if not self.dataset_ids:
            raise ValueError("no dataset images loaded")
        return self.dataset_ids[self.current_index]

    @property
    def current_annotation(self) -> HumanBeadDatasetAnnotations:
        return self.annotations[self.current_dataset_id]

    def add_dataset_image(self, dataset_id: str, path: Path, image_shape: Tuple[int, int]) -> None:
        dataset_id = str(dataset_id).strip()
        if not dataset_id:
            raise ValueError("dataset_id cannot be empty")
        self.dataset_images[dataset_id] = Path(path)
        self.image_shapes[dataset_id] = (int(image_shape[0]), int(image_shape[1]))
        if dataset_id not in self.dataset_ids:
            self.dataset_ids.append(dataset_id)
        if dataset_id not in self.annotations:
            self.annotations[dataset_id] = HumanBeadDatasetAnnotations(
                dataset_id=dataset_id,
                image_path=str(path),
                image_shape=self.image_shapes[dataset_id],
                default_radius_px=self.default_radius_px,
                beads=[],
            )

    def select_dataset(self, dataset_id: str) -> bool:
        if dataset_id not in self.dataset_ids:
            return False
        self.current_index = self.dataset_ids.index(dataset_id)
        self.selected_index = None
        return True

    def next_dataset(self) -> bool:
        if not self.dataset_ids:
            return False
        self.current_index = (self.current_index + 1) % len(self.dataset_ids)
        self.selected_index = None
        return True

    def previous_dataset(self) -> bool:
        if not self.dataset_ids:
            return False
        self.current_index = (self.current_index - 1) % len(self.dataset_ids)
        self.selected_index = None
        return True

    def _record_undo(self, dataset_id: str) -> None:
        self._undo_stack.append(
            _HistoryEntry(
                dataset_id=dataset_id,
                annotation=_clone_annotation(self.annotations[dataset_id]),
                selected_index=self.selected_index,
            )
        )
        self._redo_stack.clear()

    def _set_annotation(
        self,
        dataset_id: str,
        annotation: HumanBeadDatasetAnnotations,
        *,
        record_undo: bool = True,
    ) -> None:
        if record_undo:
            self._record_undo(dataset_id)
        self.annotations[dataset_id] = annotation
        self.dirty = True
        self.dirty_dataset_ids.add(dataset_id)

    def _next_bead_id(self, beads: Sequence[HumanBead]) -> str:
        used = {bead.bead_id for bead in beads}
        index = 1
        while True:
            bead_id = f"b{index:03d}"
            if bead_id not in used:
                return bead_id
            index += 1

    def add_bead(self, x: float, y: float) -> int:
        ann = self.current_annotation
        beads = list(ann.beads)
        beads.append(
            HumanBead(
                bead_id=self._next_bead_id(beads),
                x=float(x),
                y=float(y),
                radius_px=float(ann.default_radius_px),
            )
        )
        self._set_annotation(
            ann.dataset_id,
            HumanBeadDatasetAnnotations(ann.dataset_id, ann.image_path, ann.image_shape, ann.default_radius_px, beads),
        )
        self.selected_index = len(beads) - 1
        return self.selected_index

    def select_nearest_bead(self, x: float, y: float) -> Optional[int]:
        beads = self.current_annotation.beads
        if not beads:
            self.selected_index = None
            return None
        distances = [float(np.hypot(bead.x - float(x), bead.y - float(y))) for bead in beads]
        index = int(np.argmin(distances))
        threshold = max(6.0, 1.5 * float(beads[index].radius_px))
        self.selected_index = index if distances[index] <= threshold else None
        return self.selected_index

    def move_selected_bead(self, x: float, y: float) -> bool:
        return self.update_selected_bead(x=float(x), y=float(y))

    def update_selected_bead(
        self,
        *,
        bead_id: Optional[str] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        radius_px: Optional[float] = None,
    ) -> bool:
        if self.selected_index is None:
            return False
        ann = self.current_annotation
        if self.selected_index >= len(ann.beads):
            self.selected_index = None
            return False
        beads = list(ann.beads)
        bead = beads[self.selected_index]
        new_id = bead.bead_id if bead_id is None else str(bead_id).strip()
        if not new_id:
            raise ValueError("bead id cannot be empty")
        new_radius = bead.radius_px if radius_px is None else max(MIN_BEAD_RADIUS_PX, float(radius_px))
        beads[self.selected_index] = HumanBead(
            bead_id=new_id,
            x=bead.x if x is None else float(x),
            y=bead.y if y is None else float(y),
            radius_px=new_radius,
        )
        self._set_annotation(
            ann.dataset_id,
            HumanBeadDatasetAnnotations(ann.dataset_id, ann.image_path, ann.image_shape, ann.default_radius_px, beads),
        )
        return True

    def adjust_selected_radius(self, delta: float) -> bool:
        if self.selected_index is None:
            return False
        bead = self.current_annotation.beads[self.selected_index]
        return self.update_selected_bead(radius_px=float(bead.radius_px) + float(delta))

    def delete_selected_bead(self) -> bool:
        if self.selected_index is None:
            return False
        ann = self.current_annotation
        if self.selected_index >= len(ann.beads):
            self.selected_index = None
            return False
        beads = [bead for index, bead in enumerate(ann.beads) if index != self.selected_index]
        self._set_annotation(
            ann.dataset_id,
            HumanBeadDatasetAnnotations(ann.dataset_id, ann.image_path, ann.image_shape, ann.default_radius_px, beads),
        )
        self.selected_index = None
        return True

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        entry = self._undo_stack.pop()
        self._redo_stack.append(
            _HistoryEntry(
                dataset_id=entry.dataset_id,
                annotation=_clone_annotation(self.annotations[entry.dataset_id]),
                selected_index=self.selected_index,
            )
        )
        self.annotations[entry.dataset_id] = _clone_annotation(entry.annotation)
        self.select_dataset(entry.dataset_id)
        self.selected_index = entry.selected_index
        self.dirty = True
        self.dirty_dataset_ids.add(entry.dataset_id)
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        entry = self._redo_stack.pop()
        self._undo_stack.append(
            _HistoryEntry(
                dataset_id=entry.dataset_id,
                annotation=_clone_annotation(self.annotations[entry.dataset_id]),
                selected_index=self.selected_index,
            )
        )
        self.annotations[entry.dataset_id] = _clone_annotation(entry.annotation)
        self.select_dataset(entry.dataset_id)
        self.selected_index = entry.selected_index
        self.dirty = True
        self.dirty_dataset_ids.add(entry.dataset_id)
        return True

    def save(self) -> Path:
        write_human_bead_annotations(self.output_path, self.annotations)
        self.dirty = False
        self.dirty_dataset_ids.clear()
        return self.output_path

    def save_as(self, output_path: Path) -> Path:
        self.output_path = Path(output_path)
        return self.save()


class HumanBeadWorkbenchApp:
    """Tkinter + Matplotlib GUI for human bead annotation."""

    def __init__(
        self,
        root: Any,
        state: HumanBeadWorkbenchState,
        *,
        image_loader: Callable[[Path], np.ndarray] = _load_image,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        self.root = root
        self.state = state
        self.image_loader = image_loader
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.NavigationToolbar2Tk = NavigationToolbar2Tk
        self.Figure = Figure
        self.images: Dict[str, np.ndarray] = {}
        self.status_var = tk.StringVar(value="")
        self.progress_var = tk.StringVar(value="")
        self.selected_id_var = tk.StringVar(value="")
        self.selected_x_var = tk.StringVar(value="")
        self.selected_y_var = tk.StringVar(value="")
        self.selected_radius_var = tk.StringVar(value="")

        self.root.title("Human Bead Annotation Workbench")
        self._build_ui()
        self._load_initial_images()
        self._refresh_all()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)
        ttk.Button(top, text="Add Images", command=self._add_images).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Add Folder", command=self._add_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Save", command=self._save).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(top, text="Save As", command=self._save_as).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="Preview Current", command=self._preview_current).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(top, text="Export Previews", command=self._export_previews).pack(side=tk.LEFT, padx=2)
        ttk.Label(top, textvariable=self.status_var).pack(side=tk.RIGHT, padx=6)

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        left = ttk.Frame(paned)
        center = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(center, weight=4)
        paned.add(right, weight=1)

        ttk.Label(left, text="Datasets").pack(anchor=tk.W)
        self.dataset_tree = ttk.Treeview(
            left,
            columns=("dataset", "beads", "state"),
            show="headings",
            height=18,
            selectmode="browse",
        )
        self.dataset_tree.heading("dataset", text="Dataset")
        self.dataset_tree.heading("beads", text="Beads")
        self.dataset_tree.heading("state", text="State")
        self.dataset_tree.column("dataset", width=120, anchor=tk.W)
        self.dataset_tree.column("beads", width=60, anchor=tk.CENTER)
        self.dataset_tree.column("state", width=80, anchor=tk.W)
        self.dataset_tree.pack(fill=tk.BOTH, expand=True)
        self.dataset_tree.bind("<<TreeviewSelect>>", self._on_dataset_tree_select)
        ttk.Label(left, textvariable=self.progress_var).pack(anchor=tk.W, pady=(4, 0))

        self.figure = self.Figure(figsize=(8, 7), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.canvas = self.FigureCanvasTkAgg(self.figure, master=center)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = self.NavigationToolbar2Tk(self.canvas, center, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        ttk.Label(right, text="Beads").pack(anchor=tk.W)
        self.bead_tree = ttk.Treeview(
            right,
            columns=("id", "x", "y", "radius"),
            show="headings",
            height=10,
            selectmode="browse",
        )
        for column, width in (("id", 70), ("x", 70), ("y", 70), ("radius", 70)):
            self.bead_tree.heading(column, text=column)
            self.bead_tree.column(column, width=width, anchor=tk.CENTER)
        self.bead_tree.pack(fill=tk.BOTH, expand=False)
        self.bead_tree.bind("<<TreeviewSelect>>", self._on_bead_tree_select)

        inspector = ttk.LabelFrame(right, text="Selected")
        inspector.pack(fill=tk.X, pady=(8, 4))
        for row, (label, variable) in enumerate(
            (
                ("ID", self.selected_id_var),
                ("X", self.selected_x_var),
                ("Y", self.selected_y_var),
                ("Radius", self.selected_radius_var),
            )
        ):
            ttk.Label(inspector, text=label).grid(row=row, column=0, sticky=tk.W, padx=4, pady=2)
            ttk.Entry(inspector, textvariable=variable, width=12).grid(row=row, column=1, sticky=tk.EW, padx=4, pady=2)
        inspector.columnconfigure(1, weight=1)
        ttk.Button(inspector, text="Apply", command=self._apply_selected_values).grid(
            row=4, column=0, columnspan=2, sticky=tk.EW, padx=4, pady=(4, 2)
        )

        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Delete", command=self._delete_selected).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Undo", command=self._undo).pack(fill=tk.X, pady=2)
        ttk.Button(actions, text="Redo", command=self._redo).pack(fill=tk.X, pady=2)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-s>", lambda _event: self._save())
        self.root.bind("<Control-S>", lambda _event: self._save())
        self.root.bind("<Delete>", lambda _event: self._delete_selected())
        self.root.bind("<BackSpace>", lambda _event: self._delete_selected())
        self.root.bind("<plus>", lambda _event: self._adjust_radius(0.5))
        self.root.bind("<KP_Add>", lambda _event: self._adjust_radius(0.5))
        self.root.bind("<minus>", lambda _event: self._adjust_radius(-0.5))
        self.root.bind("<KP_Subtract>", lambda _event: self._adjust_radius(-0.5))
        self.root.bind("<Right>", lambda _event: self._go_next())
        self.root.bind("<Left>", lambda _event: self._go_previous())
        self.root.bind("n", lambda _event: self._go_next())
        self.root.bind("p", lambda _event: self._go_previous())
        self.root.bind("<Control-z>", lambda _event: self._undo())
        self.root.bind("<Control-y>", lambda _event: self._redo())

    def _load_initial_images(self) -> None:
        for dataset_id, path in self.state.dataset_images.items():
            try:
                self.images[dataset_id] = self.image_loader(Path(path))
            except Exception:
                continue

    def _refresh_all(self) -> None:
        self._refresh_dataset_tree()
        self._refresh_bead_tree()
        self._refresh_inspector()
        self._redraw_canvas()
        self._refresh_status()

    def _refresh_dataset_tree(self) -> None:
        selection = self.state.current_dataset_id if self.state.dataset_ids else ""
        self.dataset_tree.delete(*self.dataset_tree.get_children())
        for dataset_id in self.state.dataset_ids:
            ann = self.state.annotations.get(dataset_id)
            bead_count = 0 if ann is None else len(ann.beads)
            state_parts = []
            if dataset_id in self.state.dirty_dataset_ids:
                state_parts.append("dirty")
            if dataset_id not in self.images:
                state_parts.append("missing")
            self.dataset_tree.insert(
                "",
                "end",
                iid=dataset_id,
                values=(dataset_id, str(bead_count), ", ".join(state_parts) or "ok"),
            )
        if selection:
            self.dataset_tree.selection_set(selection)
            self.dataset_tree.see(selection)

    def _refresh_bead_tree(self) -> None:
        self.bead_tree.delete(*self.bead_tree.get_children())
        if not self.state.dataset_ids:
            return
        for index, bead in enumerate(self.state.current_annotation.beads):
            self.bead_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(bead.bead_id, f"{bead.x:.2f}", f"{bead.y:.2f}", f"{bead.radius_px:.2f}"),
            )
        if (
            self.state.selected_index is not None
            and self.state.selected_index < len(self.state.current_annotation.beads)
        ):
            item_id = str(self.state.selected_index)
            self.bead_tree.selection_set(item_id)
            self.bead_tree.see(item_id)

    def _refresh_inspector(self) -> None:
        if (
            not self.state.dataset_ids
            or self.state.selected_index is None
            or self.state.selected_index >= len(self.state.current_annotation.beads)
        ):
            self.selected_id_var.set("")
            self.selected_x_var.set("")
            self.selected_y_var.set("")
            self.selected_radius_var.set("")
            return
        bead = self.state.current_annotation.beads[self.state.selected_index]
        self.selected_id_var.set(bead.bead_id)
        self.selected_x_var.set(f"{bead.x:.3f}")
        self.selected_y_var.set(f"{bead.y:.3f}")
        self.selected_radius_var.set(f"{bead.radius_px:.3f}")

    def _refresh_status(self) -> None:
        if not self.state.dataset_ids:
            self.progress_var.set("No dataset images loaded")
            self.status_var.set(f"Output: {self.state.output_path}")
            return
        current = self.state.current_dataset_id
        self.progress_var.set(f"{self.state.current_index + 1}/{len(self.state.dataset_ids)}: {current}")
        suffix = " *unsaved" if self.state.dirty else ""
        self.status_var.set(f"Output: {self.state.output_path}{suffix}")

    def _redraw_canvas(self) -> None:
        self.axis.clear()
        if not self.state.dataset_ids:
            self.axis.set_title("Add images or a folder to begin")
            self.axis.set_axis_off()
            self.canvas.draw_idle()
            return
        dataset_id = self.state.current_dataset_id
        image = self.images.get(dataset_id)
        if image is None:
            self.axis.set_title(f"{dataset_id}: image unavailable")
            self.axis.set_axis_off()
            self.canvas.draw_idle()
            return
        self.axis.imshow(_robust_unit(image), cmap="gray")
        self.axis.set_title(f"{dataset_id} ({self.state.current_index + 1}/{len(self.state.dataset_ids)})")
        self.axis.set_axis_off()
        for bead_idx, bead in enumerate(self.state.current_annotation.beads):
            color = "yellow" if bead_idx == self.state.selected_index else "red"
            core = self._circle((bead.x, bead.y), bead.radius_px, color=color, linewidth=1.8)
            support = self._circle((bead.x, bead.y), 3.0 * bead.radius_px, color="cyan", linewidth=0.8)
            self.axis.add_patch(core)
            self.axis.add_patch(support)
            self.axis.plot([bead.x - 3, bead.x + 3], [bead.y, bead.y], color=color, linewidth=1)
            self.axis.plot([bead.x, bead.x], [bead.y - 3, bead.y + 3], color=color, linewidth=1)
            self.axis.text(bead.x + bead.radius_px + 2, bead.y, bead.bead_id, color=color, fontsize=8)
        self.canvas.draw_idle()

    def _circle(self, center: Tuple[float, float], radius: float, **kwargs: Any) -> Any:
        from matplotlib.patches import Circle

        return Circle(center, radius, fill=False, **kwargs)

    def _on_canvas_click(self, event: Any) -> None:
        if event.inaxes is not self.axis or event.xdata is None or event.ydata is None:
            return
        if not self.state.dataset_ids:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        selected = self.state.select_nearest_bead(x, y)
        if selected is None:
            self.state.add_bead(x, y)
        else:
            self.state.move_selected_bead(x, y)
        self._refresh_all()

    def _on_dataset_tree_select(self, _event: Any) -> None:
        selected = self.dataset_tree.selection()
        if not selected:
            return
        dataset_id = str(selected[0])
        if self.state.dataset_ids and dataset_id == self.state.current_dataset_id:
            return
        if self.state.dirty:
            self._save()
        if self.state.select_dataset(dataset_id):
            self._refresh_all()

    def _on_bead_tree_select(self, _event: Any) -> None:
        selected = self.bead_tree.selection()
        if not selected:
            return
        self.state.selected_index = int(selected[0])
        self._refresh_inspector()
        self._redraw_canvas()

    def _apply_selected_values(self) -> None:
        try:
            changed = self.state.update_selected_bead(
                bead_id=self.selected_id_var.get(),
                x=float(self.selected_x_var.get()),
                y=float(self.selected_y_var.get()),
                radius_px=float(self.selected_radius_var.get()),
            )
        except Exception as exc:
            self.messagebox.showerror("Invalid bead", str(exc))
            return
        if changed:
            self._refresh_all()

    def _delete_selected(self) -> None:
        if self.state.delete_selected_bead():
            self._refresh_all()

    def _adjust_radius(self, delta: float) -> None:
        if self.state.adjust_selected_radius(delta):
            self._refresh_all()

    def _undo(self) -> None:
        if self.state.undo():
            self._refresh_all()

    def _redo(self) -> None:
        if self.state.redo():
            self._refresh_all()

    def _go_next(self) -> None:
        if self.state.dirty:
            self._save()
        if self.state.next_dataset():
            self._refresh_all()

    def _go_previous(self) -> None:
        if self.state.dirty:
            self._save()
        if self.state.previous_dataset():
            self._refresh_all()

    def _add_images(self) -> None:
        paths = self.filedialog.askopenfilenames(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.tif *.tiff"),
                ("All Files", "*.*"),
            ]
        )
        self._add_image_paths([Path(path) for path in paths])

    def _add_folder(self) -> None:
        selected = self.filedialog.askdirectory()
        if not selected:
            return
        try:
            paths = list(collect_dataset_images_from_folder(Path(selected)).values())
        except Exception as exc:
            self.messagebox.showerror("Add Folder", str(exc))
            return
        self._add_image_paths(paths)

    def _add_image_paths(self, paths: Iterable[Path]) -> None:
        added = 0
        for path in paths:
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            try:
                image = self.image_loader(Path(path))
            except Exception as exc:
                self.messagebox.showwarning("Add Image", f"Could not load {path}: {exc}")
                continue
            dataset_id = _generated_dataset_id_for_path(Path(path), self.state.dataset_images)
            self.images[dataset_id] = image
            self.state.add_dataset_image(dataset_id, Path(path), (int(image.shape[0]), int(image.shape[1])))
            added += 1
        if added:
            self._refresh_all()

    def _save(self) -> None:
        try:
            path = self.state.save()
        except Exception as exc:
            self.messagebox.showerror("Save", str(exc))
            return
        self.status_var.set(f"Saved {path}")
        self._refresh_dataset_tree()
        self._refresh_status()

    def _save_as(self) -> None:
        selected = self.filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
            initialfile=self.state.output_path.name or "human_beads.json",
        )
        if not selected:
            return
        try:
            self.state.save_as(Path(selected))
        except Exception as exc:
            self.messagebox.showerror("Save As", str(exc))
            return
        self._refresh_all()

    def _preview_current(self) -> None:
        if not self.state.dataset_ids:
            return
        dataset_id = self.state.current_dataset_id
        image = self.images.get(dataset_id)
        if image is None:
            self.messagebox.showwarning("Preview", f"No image loaded for {dataset_id}")
            return
        output_dir = self.state.output_path.parent / "human_bead_previews"
        output = save_human_bead_preview_sheet(
            output_dir / f"{dataset_id}_human_beads.png",
            image,
            self.state.current_annotation,
        )
        self.status_var.set(f"Preview: {output}")

    def _export_previews(self) -> None:
        selected = self.filedialog.askdirectory()
        if not selected:
            return
        output_dir = Path(selected)
        written = 0
        for dataset_id, image in self.images.items():
            annotation = self.state.annotations.get(dataset_id)
            if annotation is None:
                continue
            output = save_human_bead_preview_sheet(
                output_dir / f"{dataset_id}_human_beads.png",
                image,
                annotation,
            )
            if output is not None:
                written += 1
        self.status_var.set(f"Wrote {written} preview image(s) to {output_dir}")

    def _on_close(self) -> None:
        if self.state.dirty:
            answer = self.messagebox.askyesnocancel(
                "Unsaved annotations",
                "Save annotation changes before closing?",
            )
            if answer is None:
                return
            if answer:
                self._save()
                if self.state.dirty:
                    return
        self.root.destroy()


def _initial_dataset_images_from_existing(
    output_path: Path,
    existing: Optional[Mapping[str, HumanBeadDatasetAnnotations]],
) -> Dict[str, Path]:
    if not existing:
        return {}
    result: Dict[str, Path] = {}
    for dataset_id, annotation in existing.items():
        if not annotation.image_path:
            continue
        path = Path(annotation.image_path)
        if not path.is_absolute():
            path = Path(output_path).parent / path
        if path.exists():
            result[dataset_id] = path
    return result


def launch_human_bead_workbench(
    *,
    dataset_images: Mapping[str, Path],
    output_path: Path,
    default_radius_px: float = 4.0,
    parent: Any = None,
) -> Path:
    existing = load_human_bead_annotations(Path(output_path)) if Path(output_path).exists() else None
    initial_images = dict(dataset_images)
    if not initial_images:
        initial_images = _initial_dataset_images_from_existing(Path(output_path), existing)
    image_shapes: Dict[str, Tuple[int, int]] = {}
    for dataset_id, path in initial_images.items():
        if existing is not None and dataset_id in existing:
            image_shapes[dataset_id] = existing[dataset_id].image_shape
        else:
            image_shapes[dataset_id] = _image_shape(Path(path))

    state = HumanBeadWorkbenchState(
        initial_images,
        Path(output_path),
        existing=existing,
        image_shapes=image_shapes,
        default_radius_px=default_radius_px,
    )

    import tkinter as tk

    if parent is None:
        root = tk.Tk()
        app = HumanBeadWorkbenchApp(root, state)
        setattr(root, "_human_bead_workbench_app", app)
        root.mainloop()
        return state.output_path

    top = tk.Toplevel(parent)
    app = HumanBeadWorkbenchApp(top, state)
    setattr(top, "_human_bead_workbench_app", app)
    return state.output_path
