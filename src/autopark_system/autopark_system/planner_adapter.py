"""
planner_adapter.py  —  Analytical Geometric Perpendicular Parking Planner v2
=============================================================================
Replaces the old template-search planner with a closed-form geometric solution.

COORDINATE CONVENTION (same as existing planner_core frame):
    x = depth into slot  (0 = entrance, +1.39 = wall)
    y = lateral offset   (positive = right side, car approaches from right)
    yaw = standard math  (0 = facing +x, π = facing −x = facing aisle)

ANALYTICAL PATH  (car front faces −x, yaw = π):
    R   = WB / tan(steer_max) = 1.2800 m  (steer_max = 30°)

    ① Forward LEFT setup:    gear=+1  steer=0°    dist = d1  = y_lateral + R
    ②③ Reverse 90° CCW arc:  gear=−1  steer=−30°  dist = s23 = R × π/2  (FIXED)
    ④ Reverse straight:      gear=−1  steer=0°    dist = d4  = TGT − x_depth − R
       (rear ultrasonic 6-8 stop at 40 mm is the PRIMARY stop for this move)

    Arc displacements (from yaw=π start):
        Δy_arc = +R  (rightward, centres car from y=−R to y=0)
        Δx_arc = +R  (into slot)
        Δyaw   = +π/2  (car rotates to yaw=3π/2, facing away from slot)

VALID FOR ALL THREE CASES:
    The 90° arc is entirely in the AISLE (x_depth ≤ 0), so the car never
    touches either adjacent fake car during the arc.
    During the final straight (④), the car is centred (y=0), half-width
    0.335 m < slot half-width 0.380 m — safe for both fake cars.

REQUIREMENT:
    x_depth_start ≤ −R = −1.280 m  (car must be ≥ 1.280 m into aisle)
    → Set default_start_x = −1.28 in autopark_params.yaml
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Primitive = Tuple[str, float, float]   # (direction, steer_deg, dist_m)

# ── Vehicle geometry ────────────────────────────────────────────────────────
WB        = 0.739   # wheelbase m
F_OVH     = 0.355   # front overhang m
R_OVH     = 0.170   # rear overhang m
CAR_W     = 0.670   # car width m
STEER_MAX = 30.0    # max wheel steer angle degrees

# ── Slot geometry ───────────────────────────────────────────────────────────
SLOT_W = 0.760   # slot width m
SLOT_D = 1.390   # slot depth m (entrance→wall)

# ── Stop parameters ─────────────────────────────────────────────────────────
US_REAR_STOP_M = 0.040   # rear ultrasonic sensors 6-8 stop at 40 mm from wall

# ── Derived constants ────────────────────────────────────────────────────────
R = WB / math.tan(math.radians(STEER_MAX))   # = 1.2800 m

# Target rear-axle x depth (rear bumper 40 mm from wall via ultrasonic stop)
TGT_X_AXLE = SLOT_D - R_OVH - US_REAR_STOP_M   # = 1.390 − 0.170 − 0.040 = 1.180 m

# Minimum required aisle depth before maneuver
MIN_AISLE_DEPTH = R   # = 1.280 m

# ── Compatibility dataclasses (kept from original interface) ─────────────────

@dataclass
class Pose:
    x:       float
    y:       float
    yaw_rad: float


@dataclass
class VehicleSpec:
    width:          float = CAR_W
    wheelbase:      float = WB
    front_overhang: float = F_OVH
    rear_overhang:  float = R_OVH
    max_steer_deg:  float = STEER_MAX

    @property
    def half_width(self) -> float:
        return self.width * 0.5

    @property
    def front_extent(self) -> float:
        return self.wheelbase + self.front_overhang

    @property
    def rear_extent(self) -> float:
        return self.rear_overhang


@dataclass
class PlanResult:
    success:          bool
    practical_success: bool
    strict_success:   bool
    reason:           str
    planner:          str
    case_name:        str
    primitive_seq:    List[Primitive]
    path:             List[Pose]
    metrics:          Dict[str, Any]
    score:            float


@dataclass
class PlannedPath:
    result:  PlanResult
    motions: List[Dict[str, Any]]


# ── Analytical planner ───────────────────────────────────────────────────────

def _make_straight_path(start: Pose, motions_seq: List[Tuple[int, float, float]],
                         n: int = 60) -> List[Pose]:
    """
    Simulate path for visualisation (low-resolution, not used for control).
    motions_seq: list of (gear, steer_deg, dist_m)
    """
    path = [start]
    x, y, yaw = start.x, start.y, start.yaw_rad
    for gear, steer_deg, dist_m in motions_seq:
        if dist_m <= 0:
            continue
        k = math.tan(math.radians(steer_deg)) / WB
        d = 1.0 if gear >= 0 else -1.0
        ds = dist_m / n
        for _ in range(n):
            dyaw = d * ds * k
            my   = yaw + 0.5 * dyaw
            x   += d * ds * math.cos(my)
            y   += d * ds * math.sin(my)
            yaw += dyaw
            path.append(Pose(x=x, y=y, yaw_rad=yaw))
    return path


def _analytical_plan(start_x: float, start_y: float,
                     yaw_rad: float, case_name: str) -> PlannedPath:
    """
    Core analytical planner.
    start_x = depth (should be ≤ −R)
    start_y = lateral (positive = right side)
    """
    x0_lateral = start_y    # lateral offset (right = +)
    y0_depth   = start_x    # depth position (in aisle = negative)

    # ── Validate ────────────────────────────────────────────────────────────
    if y0_depth > -MIN_AISLE_DEPTH + 0.05:
        reason = (f"aisle_too_shallow: x_depth={y0_depth:.3f} > −R={-R:.3f}  "
                  f"(car needs ≥{R:.2f}m into aisle; set default_start_x=−{R:.2f})")
        return PlannedPath(
            result=PlanResult(
                success=False, practical_success=False, strict_success=False,
                reason=reason, planner="analytical_geometric_v2",
                case_name=case_name, primitive_seq=[], path=[],
                metrics={}, score=1e9),
            motions=[]
        )

    # ── Compute distances ────────────────────────────────────────────────────
    d1  = x0_lateral + R                    # forward LEFT setup
    s23 = R * math.pi / 2                   # 90° arc (FIXED = 2.011 m)
    d4  = TGT_X_AXLE - y0_depth - R         # reverse straight into slot

    if d1 < -0.01:
        reason = (f"lateral_too_left: d1={d1:.3f} (x0={x0_lateral:.3f} < −R={-R:.3f})")
        return PlannedPath(
            result=PlanResult(
                success=False, practical_success=False, strict_success=False,
                reason=reason, planner="analytical_geometric_v2",
                case_name=case_name, primitive_seq=[], path=[],
                metrics={}, score=1e9),
            motions=[]
        )

    d1 = max(0.0, d1)
    d4 = max(0.0, d4)

    # ── Build motion sequence ────────────────────────────────────────────────
    primitives: List[Primitive] = []
    motions:    List[Dict[str, Any]] = []

    if d1 > 0.005:
        primitives.append(("f",  0.0,       d1))
        motions.append({
            "gear":        1,
            "steer_deg":   0.0,
            "dist_m":      round(d1, 4),
            "label":       "fwd_setup",
            "use_rear_us": False,
        })

    primitives.append(("r", -STEER_MAX, s23))
    motions.append({
        "gear":        -1,
        "steer_deg":   -STEER_MAX,
        "dist_m":      round(s23, 4),
        "label":       "rev_arc_90",
        "use_rear_us": False,
    })

    primitives.append(("r",  0.0,       d4))
    motions.append({
        "gear":        -1,
        "steer_deg":   0.0,
        "dist_m":      round(d4, 4),
        "label":       "rev_straight_d4",
        "use_rear_us": True,    # ← rear US sensors 6-8 stop at 40 mm
    })

    # ── Final pose (analytical, for metrics) ─────────────────────────────────
    x_final = y0_depth + R + d4          # depth: y0 + R (from arc) + d4
    y_final = 0.0                        # lateral: perfectly centred
    yaw_final = 3 * math.pi / 2          # facing away from slot

    # ── Clearances ─────────────────────────────────────────────────────────
    rear_bumper_x   = x_final + R_OVH
    rear_clear_m    = SLOT_D - rear_bumper_x
    left_clear_m    = SLOT_W / 2 - CAR_W / 2
    right_clear_m   = SLOT_W / 2 - CAR_W / 2
    yaw_err_deg     = abs(math.degrees(yaw_final - math.pi))   # 0° if perfectly aligned

    metrics = {
        "rear_clear":    rear_clear_m,
        "left_clear":    left_clear_m,
        "right_clear":   right_clear_m,
        "yaw_err_deg":   yaw_err_deg,
        "x":             x_final,
        "y":             y_final,
        "R_m":           R,
        "d1_m":          d1,
        "s23_m":         s23,
        "d4_m":          d4,
        "arc_in_aisle":  (y0_depth + R) <= 0.01,
    }

    practical_ok = (
        rear_clear_m   >= 0.015
        and left_clear_m   >= -0.030
        and right_clear_m  >= -0.030
        and yaw_err_deg    <= 2.0
    )

    # ── Simulate path for visualisation ──────────────────────────────────────
    path = _make_straight_path(
        Pose(x=start_x, y=start_y, yaw_rad=yaw_rad),
        [(m["gear"], m["steer_deg"], m["dist_m"]) for m in motions],
    )

    return PlannedPath(
        result=PlanResult(
            success=True,
            practical_success=practical_ok,
            strict_success=practical_ok,
            reason="analytical_success",
            planner="analytical_geometric_v2",
            case_name=case_name,
            primitive_seq=primitives,
            path=path,
            metrics=metrics,
            score=0.0,
        ),
        motions=motions,
    )


# ── Public interface (same as original) ──────────────────────────────────────

def plan_from_start(
    start_x:       float,
    start_y:       float,
    start_yaw_deg: float,
    case_name:     str = "right_only",
) -> PlannedPath:
    """
    Plan a perpendicular parking path.

    Args:
        start_x:       Rear-axle depth from slot entrance (m).
                       MUST be ≤ −1.280 m  (car is in the aisle).
                       Set default_start_x = −1.28 in autopark_params.yaml.
        start_y:       Rear-axle lateral offset from slot centre (m).
                       Positive = car is to the RIGHT (normal approach direction).
                       Typical value: 0.70 m.
        start_yaw_deg: Car heading (degrees).  Must be ≈ 180° (car faces −x = aisle).
        case_name:     "right_only" | "left_only" | "both_sides"
                       All three produce the same 3-move path — the analytical
                       solution avoids fake cars in all cases.

    Returns:
        PlannedPath  (compatible with result_to_dict / autopark_master)
    """
    case_name = str(case_name).strip().lower()
    if case_name not in ("left_only", "right_only", "both_sides"):
        case_name = "right_only"

    yaw_rad = math.radians(float(start_yaw_deg))
    return _analytical_plan(
        start_x=float(start_x),
        start_y=float(start_y),
        yaw_rad=yaw_rad,
        case_name=case_name,
    )


def result_to_dict(planned: PlannedPath) -> Dict[str, Any]:
    """Convert PlannedPath to dict (same interface as original)."""
    res = planned.result
    return {
        "success":           bool(res.success),
        "ok":                bool(res.success),
        "reason":            res.reason,
        "planner":           res.planner,
        "case_name":         res.case_name,
        "practical_success": bool(res.practical_success),
        "strict_success":    bool(res.strict_success),
        "maneuvers":         len(planned.motions),
        "motions":           planned.motions,
        "executable_motions": planned.motions,
        "primitive_seq":     [
            {"direction": d, "steer_deg": s, "dist_m": dist}
            for d, s, dist in res.primitive_seq
        ],
        "path":    [{"x": p.x, "y": p.y,
                     "yaw_rad": p.yaw_rad,
                     "yaw_deg": math.degrees(p.yaw_rad)} for p in res.path],
        "metrics": res.metrics,
        "score":   res.score,
    }


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"R = {R:.4f} m  (steer_max={STEER_MAX}°)")
    print(f"s23 = R×π/2 = {R*math.pi/2:.4f} m  (FIXED)")
    print(f"TGT_X_AXLE = {TGT_X_AXLE:.4f} m  (rear bumper {US_REAR_STOP_M*1000:.0f}mm from wall)")
    print()

    for case in ("right_only", "left_only", "both_sides"):
        planned = plan_from_start(-1.28, 0.70, 180.0, case)
        d = result_to_dict(planned)
        m = d.get("metrics", {})
        print(f"Case {case}:  success={d['success']}  "
              f"d1={m.get('d1_m',0):.3f}  s23={m.get('s23_m',0):.3f}  d4={m.get('d4_m',0):.3f}  "
              f"rear_clear={m.get('rear_clear',0)*1000:.0f}mm  "
              f"arc_in_aisle={m.get('arc_in_aisle')}")
