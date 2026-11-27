from __future__ import annotations

import csv
import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
SESSIONS_DIR = OUTPUTS_DIR / "sessions"
IMPORTS_FILE = OUTPUTS_DIR / "dashboard_imports.json"


def _number(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _load_session(report_path: Path) -> dict:
    with report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    csv_path = report_path.parent / "sentadilla_metricas.csv"
    rows = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    session_started_at = report.get("session_started_at")
    try:
        date_label = datetime.fromisoformat(session_started_at).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        date_label = report_path.parent.name
    return {
        "key": str(report_path.resolve()),
        "report_path": report_path,
        "date": date_label,
        "participant": report.get("participant") or "Sin identificar",
        "report": report,
        "rows": rows,
    }


def _discover_report_paths() -> list[Path]:
    paths = set(SESSIONS_DIR.glob("*/sentadilla_resumen.json"))
    legacy = OUTPUTS_DIR / "sentadilla_resumen.json"
    if legacy.is_file():
        paths.add(legacy)
    if IMPORTS_FILE.is_file():
        try:
            imported = json.loads(IMPORTS_FILE.read_text(encoding="utf-8"))
            paths.update(Path(item) for item in imported if Path(item).is_file())
        except (OSError, ValueError, TypeError):
            pass
    return sorted((path for path in paths if path.is_file()), key=lambda path: str(path))


def _person_average(person: dict, metric: str):
    return (person.get("averages") or {}).get(metric)


def _fmt(value, suffix=""):
    number = _number(value)
    return "—" if number is None else f"{number:.2f}{suffix}"


class SquatDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Pose · Seguimiento de sentadilla")
        self.geometry("1240x850")
        self.minsize(980, 680)
        self.sessions: list[dict] = []
        self.session_by_key: dict[str, dict] = {}
        self._build_ui()
        self.refresh_sessions()

    def _build_ui(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        header = ttk.Frame(self, padding=(18, 14, 18, 8))
        header.pack(fill="x")
        ttk.Label(header, text="Seguimiento de sentadilla", font=("Segoe UI", 20, "bold")).pack(side="left")
        ttk.Button(header, text="Actualizar", command=self.refresh_sessions).pack(side="right", padx=(8, 0))
        ttk.Button(header, text="Agregar reporte JSON", command=self.add_report).pack(side="right")

        self.info_label = ttk.Label(self, text="Las sesiones nuevas se guardan automáticamente en outputs/sessions.", padding=(18, 0, 18, 10))
        self.info_label.pack(fill="x")

        table_frame = ttk.Frame(self, padding=(18, 0, 18, 10))
        table_frame.pack(fill="x")
        columns = ("date", "participant", "reps", "knee", "ecc", "conc", "deviation")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6, selectmode="browse")
        headings = {
            "date": ("Fecha", 145),
            "participant": ("Persona / grupo", 150),
            "reps": ("Repeticiones", 100),
            "knee": ("Flexión máx. media", 145),
            "ecc": ("Bajada media (s)", 125),
            "conc": ("Subida media (s)", 125),
            "deviation": ("Desviación (%)", 125),
        }
        for key, (title, width) in headings.items():
            self.table.heading(key, text=title)
            self.table.column(key, width=width, anchor="center")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self.on_session_selected)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.curve_tab = ttk.Frame(self.notebook)
        self.progress_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.curve_tab, text="Curva de la sesión")
        self.notebook.add(self.progress_tab, text="Progreso entre sesiones")

        curve_controls = ttk.Frame(self.curve_tab, padding=10)
        curve_controls.pack(fill="x")
        ttk.Label(curve_controls, text="Persona detectada:").pack(side="left")
        self.person_var = tk.StringVar()
        self.person_combo = ttk.Combobox(curve_controls, textvariable=self.person_var, state="readonly", width=22)
        self.person_combo.pack(side="left", padx=8)
        self.person_combo.bind("<<ComboboxSelected>>", lambda _event: self.draw_session_curve())
        self.curve_caption = ttk.Label(curve_controls, text="Selecciona una sesión de la tabla.")
        self.curve_caption.pack(side="left", padx=12)
        self.curve_figure = Figure(figsize=(10, 6), dpi=100, constrained_layout=True)
        self.curve_canvas = FigureCanvasTkAgg(self.curve_figure, master=self.curve_tab)
        self.curve_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.curve_canvas, self.curve_tab, pack_toolbar=False).pack(side="bottom", fill="x")

        self.progress_figure = Figure(figsize=(10, 6), dpi=100, constrained_layout=True)
        self.progress_canvas = FigureCanvasTkAgg(self.progress_figure, master=self.progress_tab)
        self.progress_canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.progress_canvas, self.progress_tab, pack_toolbar=False).pack(side="bottom", fill="x")

    def refresh_sessions(self, select_key: str | None = None):
        loaded = []
        for path in _discover_report_paths():
            try:
                loaded.append(_load_session(path))
            except (OSError, ValueError, TypeError) as error:
                print(f"No se pudo leer {path}: {error}")
        self.sessions = loaded
        self.session_by_key = {session["key"]: session for session in loaded}
        for item in self.table.get_children():
            self.table.delete(item)
        for session in loaded:
            people = session["report"].get("people") or []
            if people:
                rep_count = sum(person.get("completed_repetitions", 0) for person in people)
                averages = [person.get("averages") or {} for person in people]
                max_knee = [_number(item.get("max_knee_flexion_deg")) for item in averages]
                eccentric = [_number(item.get("eccentric_duration_s")) for item in averages]
                concentric = [_number(item.get("concentric_duration_s")) for item in averages]
                deviation = [_number(item.get("target_deviation_time_pct")) for item in averages]
                average = lambda values: sum(v for v in values if v is not None) / max(1, sum(v is not None for v in values)) if any(v is not None for v in values) else None
                values = (
                    session["date"],
                    session["participant"],
                    rep_count,
                    _fmt(average(max_knee), "°"),
                    _fmt(average(eccentric)),
                    _fmt(average(concentric)),
                    _fmt(average(deviation), "%"),
                )
            else:
                values = (session["date"], session["participant"], 0, "—", "—", "—", "—")
            self.table.insert("", "end", iid=session["key"], values=values)

        if not loaded:
            self.info_label.config(text="No hay sesiones guardadas todavía. Analiza un video y vuelve a actualizar.")
        else:
            self.info_label.config(text=f"{len(loaded)} sesión(es). Selecciona una para ver sus curvas; las comparaciones agrupan por persona/grupo.")
        if select_key in self.session_by_key:
            self.table.selection_set(select_key)
            self.table.focus(select_key)
        elif loaded:
            self.table.selection_set(loaded[-1]["key"])
            self.table.focus(loaded[-1]["key"])
        self.draw_progress()

    def add_report(self):
        chosen = filedialog.askopenfilename(
            title="Seleccionar reporte de una sesión",
            initialdir=str(OUTPUTS_DIR),
            filetypes=(("Reporte JSON", "*.json"), ("Todos los archivos", "*.*")),
        )
        if not chosen:
            return
        path = Path(chosen).resolve()
        existing = []
        if IMPORTS_FILE.is_file():
            try:
                existing = json.loads(IMPORTS_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = []
        if str(path) not in existing:
            existing.append(str(path))
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        IMPORTS_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        self.refresh_sessions(select_key=str(path))

    def _selected_session(self):
        selected = self.table.selection()
        return self.session_by_key.get(selected[0]) if selected else None

    def on_session_selected(self, _event=None):
        session = self._selected_session()
        if session is None:
            return
        people = session["report"].get("people") or []
        choices = [f"ID {person.get('track_id', '?')} · {person.get('analysis_side', 'sin lado')}" for person in people]
        self.person_combo["values"] = choices
        if choices:
            self.person_combo.current(0)
        else:
            self.person_var.set("")
        self.curve_caption.config(text=f"{session['participant']} · {session['date']}")
        self.draw_session_curve()
        self.draw_progress()

    def draw_session_curve(self):
        self.curve_figure.clear()
        axes = self.curve_figure.subplots(2, 1, sharex=True)
        session = self._selected_session()
        if session is None:
            axes[0].text(0.5, 0.5, "Selecciona una sesión", ha="center", va="center", transform=axes[0].transAxes)
            axes[1].set_axis_off()
            self.curve_canvas.draw_idle()
            return
        person_index = self.person_combo.current()
        people = session["report"].get("people") or []
        if not people or person_index < 0:
            axes[0].text(0.5, 0.5, "El reporte no contiene personas/repeticiones", ha="center", va="center", transform=axes[0].transAxes)
            axes[1].set_axis_off()
            self.curve_canvas.draw_idle()
            return
        selected_track = str(people[person_index].get("track_id"))
        rows = [row for row in session["rows"] if str(row.get("track_id")) == selected_track]
        times = [_number(row.get("Tiempo (s)")) for row in rows]
        knee = [_number(row.get("knee_flexion_deg")) for row in rows]
        hip = [_number(row.get("hip_flexion_deg")) for row in rows]
        difference = [_number(row.get("trunk_shank_difference_deg")) for row in rows]
        self._plot_series(axes[0], times, ((knee, "Rodilla"), (hip, "Cadera")), "Flexión estimada", "Ángulo (°)")
        self._plot_series(axes[1], times, ((difference, "Diferencia tronco − tibia"),), "Relación tronco-tibia", "Diferencia (°)")
        axes[1].axhline(10, color="#c27b00", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[1].axhline(-10, color="#c27b00", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[1].set_xlabel("Tiempo (s)")
        self.curve_canvas.draw_idle()

    @staticmethod
    def _plot_series(axis, times, series, title, ylabel):
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.22)
        for values, label in series:
            pairs = [(x, y) for x, y in zip(times, values) if x is not None and y is not None]
            if pairs:
                axis.plot([p[0] for p in pairs], [p[1] for p in pairs], label=label, linewidth=1.5)
        if axis.lines:
            axis.legend(loc="upper right", frameon=False)
        if not any(any(x is not None and y is not None for x, y in zip(times, values)) for values, _ in series):
            axis.text(0.5, 0.5, "Sin datos de curva en el CSV", ha="center", va="center", transform=axis.transAxes)

    def draw_progress(self):
        self.progress_figure.clear()
        axes = self.progress_figure.subplots(3, 1, sharex=True)
        selected = self._selected_session()
        if selected is None:
            axes[0].text(0.5, 0.5, "No hay sesiones para comparar", ha="center", va="center", transform=axes[0].transAxes)
            axes[1].set_axis_off()
            axes[2].set_axis_off()
            self.progress_canvas.draw_idle()
            return
        participant = selected["participant"]
        sessions = (
            [selected]
            if participant == "Sin identificar"
            else [item for item in self.sessions if item["participant"] == participant]
        )
        sessions.sort(key=lambda item: item["date"])
        x = list(range(len(sessions)))
        labels = [item["date"] for item in sessions]
        metrics = []
        for item in sessions:
            people = item["report"].get("people") or []
            averages = [person.get("averages") or {} for person in people]
            def person_mean(key):
                vals = [_number(avg.get(key)) for avg in averages]
                vals = [val for val in vals if val is not None]
                return sum(vals) / len(vals) if vals else None
            metrics.append({
                "knee": person_mean("max_knee_flexion_deg"),
                "ecc": person_mean("eccentric_duration_s"),
                "conc": person_mean("concentric_duration_s"),
                "deviation": person_mean("target_deviation_time_pct"),
            })
        plots = (
            ("knee", "Flexión máxima media", "Ángulo (°)"),
            ("phase", "Duración media por fase", "Tiempo (s)"),
            ("deviation", "Tiempo con desviación del objetivo", "Porcentaje (%)"),
        )
        for axis, (metric, title, ylabel) in zip(axes, plots):
            axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.22)
            if metric == "phase":
                axis.plot(x, [item["ecc"] for item in metrics], marker="o", label="Bajada / excéntrica")
                axis.plot(x, [item["conc"] for item in metrics], marker="o", label="Subida / concéntrica")
                axis.legend(frameon=False, loc="upper right")
            else:
                key = "knee" if metric == "knee" else "deviation"
                axis.plot(x, [item[key] for item in metrics], marker="o", linewidth=1.8)
        axes[-1].set_xticks(x, labels, rotation=25, ha="right")
        if participant == "Sin identificar":
            label = "Sesión individual · usa --participant para comparar sesiones de la misma persona"
        else:
            label = f"Sesiones · {participant} (promedio de las personas detectadas en cada sesión)"
        axes[-1].set_xlabel(label)
        self.progress_canvas.draw_idle()


if __name__ == "__main__":
    app = SquatDashboard()
    app.mainloop()
