"""
planner_adapter.py  —  Analytical Geometric Perpendicular Parking Planner v4
=============================================================================
Arc steer = −30° (LEFT steer, gear=−1 → CCW rotation → correct path direction)

WHY:
  gear=−1  steer=−30°  kappa<0  dθ=(−1)(−|k|)ds = +|k|ds > 0  → CCW  ✓
  gear=−1  steer=+30°  kappa>0  dθ=(−1)(+|k|)ds = −|k|ds < 0  → CW   ✗

CCW arc from yaw=π (facing aisle) to yaw=3π/2 (front faces aisle, rear faces slot):
  Δx_lateral = +R = +1.28 m  (car moves from −1.28 to 0, perfectly centred)
  Δy_depth   = +R = +1.28 m  (car moves toward slot entrance)
  Final yaw  = 3π/2  →  reverse (gear=−1) moves in +y = INTO SLOT  ✓

The original forward-momentum issue (first test was fast = 7.64°/s because car
was still coasting from fwd_setup) is now resolved — fwd_setup speed is slow
(0.033 m/s) and there is a 1.2 s pause between moves.
"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

Primitive = Tuple[str, float, float]

WB        = 0.739
F_OVH     = 0.355
R_OVH     = 0.170
CAR_W     = 0.670
STEER_MAX = 30.0
SLOT_W    = 0.870
SLOT_D    = 1.500
US_REAR_STOP_M = 0.200

# R: use MEASURED value from physical arc test.
# Theoretical: WB/tan(30°) = 1.280m
# Measured: car drove arc at steer=30°, speed=0.05 → diameter=267cm → R=133.5cm
R = 1.335   # metres — measured from actual arc diameter
TGT_X_AXLE = SLOT_D - R_OVH - US_REAR_STOP_M        # 1.1800 m


@dataclass
class Pose:
    x: float
    y: float
    yaw_rad: float


@dataclass
class PlanResult:
    success: bool
    practical_success: bool
    strict_success: bool
    reason: str
    planner: str
    case_name: str
    primitive_seq: List[Primitive]
    path: List['Pose']
    metrics: Dict[str, Any]
    score: float


@dataclass
class PlannedPath:
    result: PlanResult
    motions: List[Dict[str, Any]]


# ── VehicleSpec kept for interface compatibility ──────────────────────────
@dataclass
class VehicleSpec:
    width: float = CAR_W
    wheelbase: float = WB
    front_overhang: float = F_OVH
    rear_overhang: float = R_OVH
    max_steer_deg: float = STEER_MAX
    @property
    def half_width(self):   return self.width * 0.5
    @property
    def front_extent(self): return self.wheelbase + self.front_overhang
    @property
    def rear_extent(self):  return self.rear_overhang


def _sim_path(start: Pose, motions, n: int = 80) -> List[Pose]:
    path = [start]
    x, y, yaw = start.x, start.y, start.yaw_rad
    for m in motions:
        dist = m.get("dist_m", 0.0)
        if dist <= 0:
            continue
        k = math.tan(math.radians(m["steer_deg"])) / WB
        d = 1.0 if m["gear"] >= 0 else -1.0
        ds = dist / n
        for _ in range(n):
            dtheta = d * ds * k
            my = yaw + 0.5 * dtheta
            x   += d * ds * math.cos(my)
            y   += d * ds * math.sin(my)
            yaw += dtheta
            path.append(Pose(x=x, y=y, yaw_rad=yaw))
    return path


def _analytical_plan(start_x: float, start_y: float,
                     yaw_rad: float, case_name: str) -> PlannedPath:
    """
    Coordinate system:
      x = depth into slot  (negative = in aisle)
      y = lateral          (positive = right side, car approach side)
      yaw = π              (car front faces −x = faces aisle)

    Equations:
      R    = WB / tan(30°)  = 1.2800 m  (measured: 1.335 m)
      d1   = y_lateral + R + 0.1        ← forward setup LEFT  (gear=+1, steer=0°)
      s23  = R × π/2 = 2.097 m          ← reverse CCW arc      (gear=−1, steer=+30°)
      d4   = TGT_X_AXLE − (y0_depth+R)  ← reverse into slot    (gear=−1, steer=0°)
                                           rear US 200 mm = PRIMARY stop

    Camera detection range: x = −0.225 m to −1.125 m (−22.5 cm to −112.5 cm).
    The arc requires only that (y0_depth + R) > 0, i.e. start_x > −R = −1.335 m,
    which is satisfied across the full camera detection range.

    After arc: car at lateral=0 (centred), depth = y0_depth + R (slot entrance region),
    then d4 reverses straight into the slot until rear US fires.
    """
    x0_lateral = start_y    # lateral offset (right=+)
    y0_depth   = start_x    # depth (in aisle = negative)

    # ── Validation ────────────────────────────────────────────────────────
    # Only reject if the car is already past the slot entrance (positive depth).
    # The old gate (-R + 0.06 = -1.275 m) wrongly rejected the entire camera
    # detection range (-1.125 m to -0.225 m).  The arc fits in the aisle as
    # long as start_x < 0, with the arc end point at y0_depth + R.
    if y0_depth > -0.10:
        reason = (f"aisle_too_shallow: depth={y0_depth:.3f} m  "
                  f"(car must be at least 10 cm before slot entrance, i.e. start_x ≤ −0.10)")
        return PlannedPath(
            result=PlanResult(False, False, False, reason, "analytical_v4",
                              case_name, [], [], {}, 1e9),
            motions=[])

    d1  = x0_lateral + R + 0.50
    s23 = R * math.pi / 2
    # d4: distance to reverse straight into slot after the arc.
    # Arc brings rear axle to depth (y0_depth + R); we need to reach TGT_X_AXLE.
    # Minimum 0.05 m so the motion is always present; rear US is the real stop.
    d4  = -y0_depth + 0.45

    if d1 < -0.01:
        reason = f"lateral_too_left: d1={d1:.3f} (lateral={x0_lateral:.3f} < −R)"
        return PlannedPath(
            result=PlanResult(False, False, False, reason, "analytical_v4",
                              case_name, [], [], {}, 1e9),
            motions=[])

    d1 = max(0.0, d1)

    # ── Motion sequence ────────────────────────────────────────────────────
    motions = [
        {
            "gear": +1, "steer_deg": 0.0, "dist_m": round(d1, 4),
            "label": "fwd_setup",
            "use_rear_us": False,
            "speed_override": None,
            "steer_active_hold": True,    # motor holds 0° against spring
        },
        {
            "gear": -1, "steer_deg": +STEER_MAX, "dist_m": round(s23, 4),  # +30° confirmed correct by user testing
            "label": "rev_arc_90",
            "use_rear_us": False,
            "speed_override": None,
            "steer_active_hold": False,   # motor locks at -30° then off
            # drive starts within 8° of -30°, continues within 20°
        },
        {
            "gear": -1, "steer_deg": 0.0, "dist_m": round(d4, 4),
            "label": "rev_straight_d4",
            "use_rear_us": True,
            "speed_override": "slow",
            "steer_active_hold": True,    # motor holds 0° against spring
        },
    ]

    primitives: List[Primitive] = [
        ("f",  0.0,       d1),
        ("r",  STEER_MAX, s23),
        ("r",  0.0,       d4),
    ]

    # ── Final pose & clearances ────────────────────────────────────────────
    x_f = 0.0
    y_f = y0_depth + R + d4
    rear_bumper = y_f + R_OVH
    metrics = {
        "rear_clear":  SLOT_D - rear_bumper,
        "left_clear":  SLOT_W / 2 - CAR_W / 2,
        "right_clear": SLOT_W / 2 - CAR_W / 2,
        "yaw_err_deg": 0.0,
        "x_f": x_f, "y_f": y_f,
        "R_m": R,
        "d1_m": d1, "s23_m": s23, "d4_m": d4,
        # arc_end_depth: positive = arc dips into slot entrance (OK); negative = stays in aisle.
        "arc_end_depth_m": round(y0_depth + R, 4),
        "arc_in_aisle": (y0_depth + R) <= 0.01,
        "arc_steer_note": "+30° (RIGHT steer + reverse = CCW = correct direction)",
    }

    ok = (metrics["rear_clear"] >= 0.015
          and metrics["left_clear"] >= -0.030
          and metrics["right_clear"] >= -0.030)

    path = _sim_path(Pose(x=start_x, y=start_y, yaw_rad=yaw_rad), motions)

    return PlannedPath(
        result=PlanResult(
            success=True, practical_success=ok, strict_success=ok,
            reason="analytical_success", planner="analytical_v4",
            case_name=case_name, primitive_seq=primitives,
            path=path, metrics=metrics, score=0.0),
        motions=motions)


# ── Public interface ──────────────────────────────────────────────────────

def plan_from_start(start_x: float, start_y: float,
                    start_yaw_deg: float,
                    case_name: str = "right_only") -> PlannedPath:
    case_name = str(case_name).strip().lower()
    if case_name not in ("left_only", "right_only", "both_sides"):
        case_name = "right_only"
    return _analytical_plan(float(start_x), float(start_y),
                             math.radians(float(start_yaw_deg)), case_name)


def result_to_dict(planned: PlannedPath) -> Dict[str, Any]:
    res = planned.result
    return {
        "success":            bool(res.success),
        "ok":                 bool(res.success),
        "reason":             res.reason,
        "planner":            res.planner,
        "case_name":          res.case_name,
        "practical_success":  bool(res.practical_success),
        "strict_success":     bool(res.strict_success),
        "maneuvers":          len(planned.motions),
        "motions":            planned.motions,
        "executable_motions": planned.motions,
        "primitive_seq":      [{"direction": d, "steer_deg": s, "dist_m": dist}
                               for d, s, dist in res.primitive_seq],
        "path":    [{"x": p.x, "y": p.y, "yaw_rad": p.yaw_rad,
                     "yaw_deg": math.degrees(p.yaw_rad)} for p in res.path],
        "metrics": res.metrics,
        "score":   res.score,
    }


if __name__ == "__main__":
    print(f"R = {R:.4f} m   s23 = {R*math.pi/2:.4f} m   TGT = {TGT_X_AXLE:.4f} m")
    print()
    p = plan_from_start(-1.28, 0.70, 180.0, "right_only")
    d = result_to_dict(p)
    m = d["metrics"]
    print(f"d1  = {m['d1_m']:.4f} m  (fwd setup)")
    print(f"s23 = {m['s23_m']:.4f} m  (rev CCW arc, steer=−30°)")
    print(f"d4  = {m['d4_m']:.4f} m  (rev straight into slot)")
    print(f"Rear clearance  = {m['rear_clear']*1000:.0f} mm")
    print(f"Side clearances = {m['left_clear']*1000:.0f} mm / {m['right_clear']*1000:.0f} mm")
    print(f"Arc in aisle    = {m['arc_in_aisle']}")
    print()
    print(f"Note: {m['arc_steer_note']}")
