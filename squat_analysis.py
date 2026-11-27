"""Metricas 2D de sentadilla lateral y resumen por repeticion."""

from __future__ import annotations

from collections import defaultdict, deque
from math import acos, atan2, degrees, hypot
from statistics import mean, median


MIN_CONFIDENCE = 0.35
SMOOTHING_WINDOW = 5
LANDMARKS_PER_SIDE = ("shoulder", "hip", "knee", "ankle")
POINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

CSV_COLUMNS = [
    "frame_index",
    "Tiempo (s)",
    "track_id",
    "analysis_side",
    "side_confidence_mean",
    "phase",
    "rep_count",
    "rep_id",
    "knee_flexion_deg",
    "hip_flexion_deg",
    "trunk_inclination_deg",
    "shank_inclination_deg",
    "trunk_shank_difference_deg",
    "actual_bias",
    "target_bias",
    "posture_alert",
    "posture_label",
    "posture_hint",
    "hip_drop_percent_leg_length",
    "depth_category",
    "rep_max_knee_flexion_deg",
    "eccentric_duration_s",
    "concentric_duration_s",
    "target_deviation_time_pct",
]

DEPTH_CATEGORIES = (
    (90.0, "parcial"),
    (110.0, "media"),
    (135.0, "profunda"),
    (float("inf"), "fuera_de_rango_referencia"),
)


def _point(xy, confidence, name, min_confidence):
    index = POINTS[name]
    if confidence is not None and float(confidence[index]) < min_confidence:
        return None
    return float(xy[index][0]), float(xy[index][1])


def _angle(a, vertex, c):
    if a is None or vertex is None or c is None:
        return None
    va = (a[0] - vertex[0], a[1] - vertex[1])
    vc = (c[0] - vertex[0], c[1] - vertex[1])
    denominator = hypot(*va) * hypot(*vc)
    if denominator < 1e-6:
        return None
    cosine = (va[0] * vc[0] + va[1] * vc[1]) / denominator
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _flexion(a, vertex, c):
    angle = _angle(a, vertex, c)
    return None if angle is None else 180.0 - angle


def _average_confidence(side, confidence, points):
    indices = [POINTS[f"{side}_{joint}"] for joint in LANDMARKS_PER_SIDE]
    if confidence is not None:
        return mean(float(confidence[index]) for index in indices)
    visible = sum(points[f"{side}_{joint}"] is not None for joint in LANDMARKS_PER_SIDE)
    return visible / len(indices)


class SquatAnalyzer:
    """Analiza un lado fijo por persona tras calibrarlo con las primeras poses.

    La vista debe ser lateral, fija y perpendicular al movimiento. El indicador
    tronco-tibia clasifica sesgo biomecanico; no diagnostica una tecnica universal.
    """

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        smoothing_window: int = SMOOTHING_WINDOW,
        calibration_frames: int = 15,
        rep_start_flexion_deg: float = 45.0,
        standing_flexion_deg: float = 15.0,
        target_bias: str = "neutral",
        bias_tolerance_deg: float = 10.0,
        alert_min_knee_flexion_deg: float = 90.0,
        max_gap_seconds: float = 2.0,
    ) -> None:
        if target_bias not in {"neutral", "hip", "knee"}:
            raise ValueError("target_bias debe ser 'neutral', 'hip' o 'knee'")
        self.min_confidence = min_confidence
        self.calibration_frames = max(1, calibration_frames)
        self.rep_start_flexion_deg = rep_start_flexion_deg
        self.standing_flexion_deg = standing_flexion_deg
        self.target_bias = target_bias
        self.bias_tolerance_deg = bias_tolerance_deg
        self.alert_min_knee_flexion_deg = alert_min_knee_flexion_deg
        self.max_gap_seconds = max_gap_seconds
        self._calibration = {}
        self._locked_side = {}
        self._locked_confidence = {}
        self._history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=smoothing_window)))
        self._state = defaultdict(
            lambda: {
                "phase": "calibrando_lado",
                "rep_count": 0,
                "current_rep": None,
                "baseline_hip_y": None,
                "leg_length": None,
                "previous_time": None,
            }
        )
        self._completed_reps = defaultdict(list)

    def _calibrate_side(self, track_id, points, confidence):
        if track_id in self._locked_side:
            return self._locked_side[track_id]

        state = self._calibration.setdefault(
            track_id,
            {"count": 0, "sums": {"left": 0.0, "right": 0.0}},
        )
        if state["count"] < self.calibration_frames:
            for side in ("left", "right"):
                state["sums"][side] += _average_confidence(side, confidence, points)
            state["count"] += 1

        if state["count"] >= self.calibration_frames:
            averages = {
                side: value / state["count"] for side, value in state["sums"].items()
            }
            selected_side = max(averages, key=averages.get)
            self._locked_side[track_id] = selected_side
            self._locked_confidence[track_id] = averages[selected_side]
            return self._locked_side[track_id]
        return None

    def _classify_bias(self, difference):
        if difference > self.bias_tolerance_deg:
            return "hip"
        if difference < -self.bias_tolerance_deg:
            return "knee"
        return "neutral"

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        track_id: int | None,
        keypoints_xy,
        keypoints_conf=None,
    ) -> dict[str, int | float | str | bool | None]:
        row = {column: None for column in CSV_COLUMNS}
        row.update(
            frame_index=frame_index,
            track_id=track_id,
            phase="sin_ID" if track_id is None else "calibrando_lado",
            target_bias=self.target_bias,
            posture_alert=False,
        )
        row["Tiempo (s)"] = round(timestamp_s, 6)
        if track_id is None:
            return row

        points = {
            name: _point(keypoints_xy, keypoints_conf, name, self.min_confidence)
            for name in POINTS
        }
        side = self._calibrate_side(track_id, points, keypoints_conf)
        if side is None:
            return row

        row["analysis_side"] = side
        row["side_confidence_mean"] = round(self._locked_confidence[track_id], 4)
        shoulder = points[f"{side}_shoulder"]
        hip = points[f"{side}_hip"]
        knee = points[f"{side}_knee"]
        ankle = points[f"{side}_ankle"]
        if shoulder is None or hip is None or knee is None or ankle is None:
            state = self._state[track_id]
            row["phase"] = state["phase"]
            row["rep_count"] = state["rep_count"]
            return row

        row["knee_flexion_deg"] = _flexion(hip, knee, ankle)
        row["hip_flexion_deg"] = _flexion(shoulder, hip, knee)

        facing_sign = 1.0
        nose_confidence = float(keypoints_conf[0]) if keypoints_conf is not None else 1.0
        if nose_confidence >= self.min_confidence:
            nose_x = float(keypoints_xy[0][0])
            if abs(nose_x - shoulder[0]) > 1.0:
                facing_sign = 1.0 if nose_x > shoulder[0] else -1.0
        trunk_angle = degrees(atan2(shoulder[0] - hip[0], hip[1] - shoulder[1]))
        shank_angle = degrees(atan2(knee[0] - ankle[0], ankle[1] - knee[1]))
        row["trunk_inclination_deg"] = trunk_angle * facing_sign
        row["shank_inclination_deg"] = shank_angle * facing_sign
        difference = row["trunk_inclination_deg"] - row["shank_inclination_deg"]
        row["trunk_shank_difference_deg"] = difference

        state = self._state[track_id]
        smooth_metrics = (
            "knee_flexion_deg",
            "hip_flexion_deg",
            "trunk_inclination_deg",
            "shank_inclination_deg",
            "trunk_shank_difference_deg",
        )
        for metric in smooth_metrics:
            value = row[metric]
            if value is not None:
                self._history[track_id][metric].append(float(value))
                row[metric] = float(median(self._history[track_id][metric]))

        knee_flexion = row["knee_flexion_deg"]
        hip_flexion = row["hip_flexion_deg"]
        trunk_shank = row["trunk_shank_difference_deg"]
        row["actual_bias"] = self._classify_bias(trunk_shank)
        if knee_flexion is not None:
            row["depth_category"] = next(
                category
                for upper_bound, category in DEPTH_CATEGORIES
                if knee_flexion < upper_bound
            )

        actual_bias = row["actual_bias"]
        if (
            knee_flexion is not None
            and knee_flexion >= self.alert_min_knee_flexion_deg
            and actual_bias != self.target_bias
        ):
            row["posture_alert"] = True
            if self.target_bias == "neutral":
                row["posture_label"] = (
                    "DOMINANCIA DE CADERA" if actual_bias == "hip" else "DOMINANCIA DE RODILLA"
                )
            else:
                target_name = {"hip": "CADERA", "knee": "RODILLA"}[self.target_bias]
                row["posture_label"] = f"DESVIACION DEL OBJETIVO {target_name}"
            if actual_bias == "hip":
                row["posture_hint"] = "Tronco inclina mas que tibia; diferencia positiva"
            elif actual_bias == "knee":
                row["posture_hint"] = "Tibia inclina mas que tronco; diferencia negativa"
        else:
            row["posture_label"] = "OBJETIVO NEUTRAL" if actual_bias == "neutral" else f"SESGO {actual_bias.upper()}"

        current_rep = state["current_rep"]
        previous_time = state["previous_time"]
        dt = timestamp_s - previous_time if previous_time is not None else 0.0
        if dt < 0 or dt > self.max_gap_seconds:
            dt = 0.0
        alert_now = bool(row["posture_alert"])

        if current_rep is None and knee_flexion is not None and knee_flexion >= self.rep_start_flexion_deg:
            current_rep = {
                "rep_id": state["rep_count"] + 1,
                "start_time_s": timestamp_s,
                "bottom_time_s": timestamp_s,
                "max_knee_flexion_deg": knee_flexion,
                "duration_observed_s": 0.0,
                "warning_duration_s": 0.0,
                "phase": "descenso",
            }
            state["current_rep"] = current_rep

        if current_rep is not None:
            current_rep["duration_observed_s"] += dt
            if alert_now:
                current_rep["warning_duration_s"] += dt
            if knee_flexion is not None and knee_flexion > current_rep["max_knee_flexion_deg"]:
                current_rep["max_knee_flexion_deg"] = knee_flexion
                current_rep["bottom_time_s"] = timestamp_s

            if current_rep["phase"] == "descenso":
                if knee_flexion is not None and knee_flexion < current_rep["max_knee_flexion_deg"] - 5.0:
                    current_rep["phase"] = "fondo"
            elif current_rep["phase"] == "fondo":
                if knee_flexion is not None and knee_flexion < current_rep["max_knee_flexion_deg"] - 5.0:
                    current_rep["phase"] = "ascenso"

            if (
                current_rep["phase"] == "ascenso"
                and knee_flexion is not None
                and knee_flexion <= self.standing_flexion_deg
            ):
                eccentric = max(0.0, current_rep["bottom_time_s"] - current_rep["start_time_s"])
                concentric = max(0.0, timestamp_s - current_rep["bottom_time_s"])
                observed = current_rep["duration_observed_s"]
                bad_pct = (
                    100.0 * current_rep["warning_duration_s"] / observed
                    if observed > 0
                    else 0.0
                )
                rep_summary = {
                    "rep_id": current_rep["rep_id"],
                    "start_time_s": round(current_rep["start_time_s"], 4),
                    "end_time_s": round(timestamp_s, 4),
                    "eccentric_duration_s": round(eccentric, 4),
                    "concentric_duration_s": round(concentric, 4),
                    "max_knee_flexion_deg": round(current_rep["max_knee_flexion_deg"], 2),
                    "target_deviation_time_pct": round(min(100.0, bad_pct), 2),
                    "target_bias": self.target_bias,
                }
                self._completed_reps[track_id].append(rep_summary)
                state["rep_count"] += 1
                row["eccentric_duration_s"] = rep_summary["eccentric_duration_s"]
                row["concentric_duration_s"] = rep_summary["concentric_duration_s"]
                row["rep_max_knee_flexion_deg"] = rep_summary["max_knee_flexion_deg"]
                row["target_deviation_time_pct"] = rep_summary["target_deviation_time_pct"]
                state["current_rep"] = None
                current_rep = None
                state["phase"] = "de_pie"
            else:
                state["phase"] = current_rep["phase"]

        if knee_flexion is not None and hip_flexion is not None:
            if state["baseline_hip_y"] is None and knee_flexion <= self.standing_flexion_deg:
                state["baseline_hip_y"] = hip[1]
                state["leg_length"] = max(1.0, hypot(hip[0] - ankle[0], hip[1] - ankle[1]))
            if state["baseline_hip_y"] is not None and state["leg_length"]:
                row["hip_drop_percent_leg_length"] = round(
                    100.0 * max(0.0, hip[1] - state["baseline_hip_y"]) / state["leg_length"],
                    2,
                )

        state["previous_time"] = timestamp_s
        row["phase"] = state["phase"]
        row["rep_count"] = state["rep_count"]
        current_rep = state["current_rep"]
        row["rep_id"] = current_rep["rep_id"] if current_rep is not None else state["rep_count"]
        if current_rep is not None:
            row["rep_max_knee_flexion_deg"] = round(current_rep["max_knee_flexion_deg"], 2)
        return row

    def summary(self, frame_count: int, fps: float, time_basis: str) -> dict:
        people = []
        for track_id, side in self._locked_side.items():
            reps = self._completed_reps.get(track_id, [])
            people.append(
                {
                    "track_id": track_id,
                    "analysis_side": side,
                    "completed_repetitions": len(reps),
                    "averages": {
                        "eccentric_duration_s": round(
                            mean(rep["eccentric_duration_s"] for rep in reps), 4
                        ) if reps else None,
                        "concentric_duration_s": round(
                            mean(rep["concentric_duration_s"] for rep in reps), 4
                        ) if reps else None,
                        "max_knee_flexion_deg": round(
                            mean(rep["max_knee_flexion_deg"] for rep in reps), 2
                        ) if reps else None,
                        "target_deviation_time_pct": round(
                            mean(rep["target_deviation_time_pct"] for rep in reps), 2
                        ) if reps else None,
                    },
                    "repetitions": reps,
                }
            )
        return {
            "time_basis": time_basis,
            "fps_used_for_timing": fps,
            "frame_count": frame_count,
            "side_calibration_frames": self.calibration_frames,
            "target_bias": self.target_bias,
            "people": people,
            "notes": [
                "Solo se incluyen repeticiones completas; las fases se estiman con angulos 2D.",
                "El porcentaje es tiempo observado fuera del objetivo de sesgo, no tiempo de lesion ni diagnostico.",
            ],
        }
