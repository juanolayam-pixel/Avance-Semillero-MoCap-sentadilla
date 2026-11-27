"""Calculos cinematicos horizontales 2D para marcha lineal."""

from __future__ import annotations

from collections import defaultdict, deque
from statistics import median


POINTS = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
ANALYSIS_POINTS = ("hip_center", *POINTS.keys())
DEFAULT_CONFIDENCE = 0.35
DEFAULT_SMOOTHING_WINDOW = 5
DEFAULT_MAX_GAP_SECONDS = 0.5


def csv_columns() -> list[str]:
    """Columnas del registro por persona y fotograma."""
    columns = ["frame_index", "timestamp_s", "track_id"]
    for point_name in ANALYSIS_POINTS:
        columns.extend(
            (
                f"{point_name}_x_raw_px",
                f"{point_name}_y_raw_px",
                f"{point_name}_confidence",
                f"{point_name}_x_px",
                f"{point_name}_y_px",
                f"{point_name}_dx_px",
                f"{point_name}_velocity_x_px_s",
                f"{point_name}_speed_x_px_s",
                f"{point_name}_displacement_x_px",
                f"{point_name}_distance_x_px",
            )
        )
    return columns


class HorizontalKinematics:
    """Deriva posicion, desplazamiento, distancia y velocidad en el eje X.

    Las posiciones se conservan en pixeles. Para estimar metros se necesita
    calibrar la escala de la imagen en el plano de marcha.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_gap_seconds = max_gap_seconds
        self._history: dict[tuple[int, str], deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=smoothing_window)
        )
        self._previous: dict[tuple[int, str], tuple[float, float]] = {}
        self._origin: dict[tuple[int, str], float] = {}
        self._distance: dict[tuple[int, str], float] = defaultdict(float)

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        track_id: int | None,
        keypoints_xy,
        keypoints_conf=None,
    ) -> dict[str, int | float | str | None]:
        """Devuelve una fila CSV para una deteccion y su instante de video."""
        row: dict[str, int | float | str | None] = {
            "frame_index": frame_index,
            "timestamp_s": round(timestamp_s, 6),
            "track_id": track_id,
        }

        samples: dict[str, tuple[float, float, float] | None] = {}
        for name, index in POINTS.items():
            conf = self._confidence(keypoints_conf, index)
            x, y = float(keypoints_xy[index][0]), float(keypoints_xy[index][1])
            samples[name] = (x, y, conf) if conf >= self.confidence_threshold else None

        visible_hips = [samples[name] for name in ("left_hip", "right_hip") if samples[name]]
        if visible_hips:
            hip_center = (
                sum(sample[0] for sample in visible_hips) / len(visible_hips),
                sum(sample[1] for sample in visible_hips) / len(visible_hips),
                sum(sample[2] for sample in visible_hips) / len(visible_hips),
            )
        else:
            hip_center = None
        samples["hip_center"] = hip_center

        for name in ANALYSIS_POINTS:
            sample = samples[name]
            raw_x, raw_y, confidence = sample if sample is not None else (None, None, None)
            row[f"{name}_x_raw_px"] = raw_x
            row[f"{name}_y_raw_px"] = raw_y
            row[f"{name}_confidence"] = confidence

            metric = self._derive(name, track_id, timestamp_s, sample)
            for key, value in metric.items():
                row[f"{name}_{key}"] = value
        return row

    def _derive(
        self,
        name: str,
        track_id: int | None,
        timestamp_s: float,
        sample: tuple[float, float, float] | None,
    ) -> dict[str, float | None]:
        fields = {
            "x_px": None,
            "y_px": None,
            "dx_px": None,
            "velocity_x_px_s": None,
            "speed_x_px_s": None,
            "displacement_x_px": None,
            "distance_x_px": None,
        }
        if sample is None:
            return fields

        x_raw, y_raw, _ = sample
        # Sin ID persistente no mezclamos el historial de personas distintas.
        if track_id is None:
            fields["x_px"], fields["y_px"] = x_raw, y_raw
            return fields

        key = (track_id, name)
        history = self._history[key]
        history.append((x_raw, y_raw))
        x = float(median(point[0] for point in history))
        y = float(median(point[1] for point in history))
        fields["x_px"], fields["y_px"] = x, y

        previous = self._previous.get(key)
        if previous is None:
            self._origin[key] = x
            self._distance[key] = 0.0
        else:
            previous_time, previous_x = previous
            dt = timestamp_s - previous_time
            if 0 < dt <= self.max_gap_seconds:
                dx = x - previous_x
                velocity = dx / dt
                self._distance[key] += abs(dx)
                fields["dx_px"] = dx
                fields["velocity_x_px_s"] = velocity
                fields["speed_x_px_s"] = abs(velocity)

        self._previous[key] = (timestamp_s, x)
        fields["displacement_x_px"] = x - self._origin[key]
        fields["distance_x_px"] = self._distance[key]
        return fields

    @staticmethod
    def _confidence(keypoints_conf, index: int) -> float:
        if keypoints_conf is None:
            return 1.0
        return float(keypoints_conf[index])
