"""
planner_adapter.py  —  Analytical Geometric Perpendicular Parking Planner v6
=============================================================================
Arc steer = −30° (LEFT steer, gear=−1 → CCW rotation → correct path direction)

WHY:
  gear=−1  steer=−30°  kappa<0  dθ=(−1)(−|k|)ds = +|k|ds > 0  → CCW  ✓
  gear=−1  steer=+30°  kappa>0  dθ=(−1)(+|k|)ds = −|k|ds < 0  → CW   ✗

CCW arc from yaw=π (facing aisle) to yaw=3π/2 (front faces aisle, rear faces slot):
  Δx_lateral = +R = +1.335 m  (car moves from −R to 0, centred in slot)
  Δy_depth   = +R = +1.335 m  (car moves toward slot entrance)
  Final yaw  = 3π/2  →  reverse (gear=−1) moves in +y = INTO SLOT  ✓

v6 changes (for 1-second replan loop + rear-cam + US centering architecture):
  Move 1 now has TWO paths selected by how far the car is from the slot entrance:

    x0_aisle >= R  →  STRAIGHT  (steer1 = 0°)
      Car is R or more from slot. Arc has natural room; no pre-angle needed.
      steer = 0° = spring-return default, so steer_ready fires almost instantly.
      Saves ~2 s of steer-settle time vs the 30° path.
      d1 = R_MIN_STEER × |theta_1| (same short formula as the steer path).

    x0_aisle < R   →  MAXSTEER  (steer1 = −30°)
      Car is less than R from slot. Max steer pre-angles in minimum distance.
      d1 = R_MIN_STEER × |theta_1| ≈ 0.30 m  (86 % shorter than old ~2.2 m).
      Lateral residual (~22 cm) is corrected by rear_cam + US4/US5 centering.

  Arc (s23) and Move 3 (d4) formulas are identical for both paths.
  d4 fallback buffer reduced from +0.40 m to +0.20 m."""
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
# User measured: R = 167cm = 1.670m
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

    Equations (v6 — two paths for Move 1):

      x0_aisle = −start_x  (distance from slot entrance, always positive)

      IF x0_aisle >= R  (car is far from slot → STRAIGHT Move 1):
        d1      = R_MIN_STEER × |theta_1|   steer1 = 0°
      ELSE           (car is close to slot → MAXSTEER Move 1):
        d1      = R_MIN_STEER × |theta_1|   steer1 = −30°

      s23     = R × (π/2 − theta_m1)    ← reverse CCW arc  (same for both paths)
      d4      = −y0_depth + 0.20        ← reverse into slot (rear_cam overrides)

    STRAIGHT path: steer=0° is the spring-return default; steer_ready fires
    almost instantly, saving ~2 s of settle time.
    MAXSTEER path: pre-angles the car in ~0.30 m before the arc; residual
    lateral error is corrected by rear_cam (yaw) and US4/US5 (lateral).
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
            result=PlanResult(False, False, False, reason, "analytical_v5",
                              case_name, [], [], {}, 1e9),
            motions=[])

    # ── Move 1 decision ───────────────────────────────────────────────────
    # R_MIN_STEER = WB / tan(30°) = 1.280 m — the tightest the steering can turn.
    # x0_aisle = distance the car is from the slot entrance (always positive).
    #
    # Two paths based on how far the car is from the slot:
    #
    #   x0_aisle >= R  →  STRAIGHT path (steer1 = 0°)
    #     The car is R or more from the slot entrance.  The arc has natural
    #     room; no pre-angle is needed.  steer=0° is the spring-return default,
    #     so steer_ready fires almost instantly — saves ~2 s of settle time vs 30°.
    #     d1 uses the same short formula as the steer path.
    #
    #   x0_aisle < R   →  MAXSTEER path (steer1 = −30°)
    #     Car is less than R from the slot entrance.  Max steer pre-angles the
    #     car in the minimum distance (~0.30 m) before the arc.
    #     Residual lateral error (~22 cm) is corrected by rear_cam + US centering.
    #
    # Both paths use d1 = R_MIN_STEER × |theta_1| (same short formula).
    # The arc (s23) and d4 are identical in both cases.
    R_MIN_STEER = WB / math.tan(math.radians(STEER_MAX))   # 1.280 m

    x0_aisle   = -y0_depth                      # aisle depth (always positive)
    theta_m1   = math.atan2(R - x0_lateral, x0_aisle + R)
    theta_0    = yaw_rad - math.pi               # yaw deviation from π (ideal = 0)
    theta_1    = theta_m1 - theta_0             # yaw change needed in Move 1

    if x0_aisle >= R:
        # ── STRAIGHT path: car is far enough, no steer needed ────────────
        r1         = R_MIN_STEER
        d1         = max(r1 * abs(theta_1), 0.10)
        steer1_deg = 0.0                         # 0° = spring default, settles instantly

    elif abs(theta_1) < math.radians(1.0):
        # ── Nearly aligned and close — short straight run ─────────────────
        d1_geo     = x0_lateral + R + 0.05
        d1         = max(d1_geo, 0.10)
        steer1_deg = 0.0

    else:
        # ── MAXSTEER path: car is close (x0_aisle < R), pre-angle needed ──
        r1         = R_MIN_STEER
        d1         = max(r1 * abs(theta_1), 0.10)
        steer1_deg = -STEER_MAX                  # −30° → steer_ready fires after settle

    d1  = max(d1, 0.10)
    # Move 2 arc distance — user formula (from diagram):
    # s23 = πR × (90° − θm1) / 180°  =  R × (π/2 − θm1)
    # Move 1 pre-angles the car by θm1, so arc only covers the remaining
    # (90° − θm1) degrees to complete the 90° slot-entry turn.
    # When θm1=0° (no pre-angle): s23 = R×π/2 = 2.097m (original value).
    # When θm1=15.9° (y=0.70): s23 = 1.727m (shorter arc). ✓
    s23 = R * (math.pi / 2 - theta_m1)   # = πR × (90-θm1_deg) / 180
    s23 = max(s23, 0.10)  # safety: never negative
    d4  = -y0_depth + 0.20   # fallback only — rear_cam overrides at runtime
                              # reduced buffer (0.40→0.20) to avoid over-travel
                              # if rear_cam fails

    if d1 < -0.01:
        reason = f"lateral_too_left: d1={d1:.3f} (lateral={x0_lateral:.3f} < −R)"
        return PlannedPath(
            result=PlanResult(False, False, False, reason, "analytical_v5",
                              case_name, [], [], {}, 1e9),
            motions=[])

    # ── Motion sequence ────────────────────────────────────────────────────
    motions = [
        {
            "gear": +1, "steer_deg": round(steer1_deg, 2), "dist_m": round(d1, 4),
            "label": "fwd_setup",
            "use_rear_us": False,
            "speed_override": None,
            "steer_active_hold": True,    # motor holds 0° against spring
            "no_imu_correct": True,       # Move1: steer_ready then drive straight, no IMU correction
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
            # ── Move 3: rear-camera-guided reverse into slot ───────────────
            # steer_deg and dist_m below are PLANNER ESTIMATES used as fallbacks
            # only if the rear camera times out. At runtime autopark_master
            # overrides both values using the rear camera reading taken between
            # Move 2 and Move 3 (see autopark_master._compute_move3_from_rear_cam):
            #
            #   y_rear   = dist from rear cam to slot back wall (m)
            #   yaw_rear = tilt angle car vs slot axis (deg)
            #   a        = y_rear · cos(yaw_rear)           # depth component
            #   x_rear   = y_rear · sin(yaw_rear)           # lateral component
            #   d3       = sqrt((a − 0.375)² + x_rear²)     # Euclidean drive dist
            #   steer    = −yaw_rear                         # correct misalignment
            #
            # no_imu_correct=True keeps is_turning=False even when steer≠0,
            # so the encoder (not IMU arc) remains the stop condition.
            "gear": -1, "steer_deg": 0.0, "dist_m": round(d4, 4),
            "label": "rev_straight_d4",
            "use_rear_cam": True,          # triggers rear-cam pre-measure in autopark_master
            "use_encoder":  True,          # forces encoder intercept even when steer≠0
            "use_rear_us": False,          # encoder stop — rear US not used
            "steer_active_hold": True,     # motor holds steer against spring return
            "no_imu_correct": True,        # Move3: steer_ready then drive, encoder stops
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
        "d1_m": d1, "steer1_deg": round(steer1_deg, 2),
        "theta1_deg": round(math.degrees(theta_1), 2),
        "s23_m": s23,
        "theta_m1_deg": round(math.degrees(theta_m1), 2),
        "d4_m": d4,
        "d4_fallback_m": round(d4, 4),
        # arc_end_depth: positive = arc dips into slot entrance (OK).
        "arc_end_depth_m": round(y0_depth + R, 4),
        "arc_in_aisle": (y0_depth + R) <= 0.01,
        "arc_steer_note": "+30° (RIGHT steer + reverse = CCW = correct direction)",
        "move3_rear_cam": True,
        # ── v6 formula info ──────────────────────────────────────────────
        # Records which Move 1 path was selected.
        "move1_path":    "straight" if steer1_deg == 0.0 else "maxsteer",
        "move1_reason":  (
            "x0_aisle>=R: straight (no pre-angle needed)"  if x0_aisle >= R
            else "theta1<1deg: nearly aligned"             if abs(theta_1) < math.radians(1.0)
            else "x0_aisle<R: maxsteer (pre-angle required)"
        ),
        # Rough time estimates at typical speeds (for 20-second target tracking).
        # Actual times depend on encoder_bridge speeds and YAML tuning.
        "est_d1_time_s":  round(d1  / 0.10, 2),   # Move 1 @ 0.10 m/s
        "est_arc_time_s": round(s23 / 0.236, 2),   # Arc    @ 0.236 m/s (physics limit)
        "est_d4_time_s":  round(d4  / 0.15, 2),    # Move 3 @ 0.15 m/s (tuned d4_speed)
    }

    ok = (metrics["rear_clear"] >= 0.015
          and metrics["left_clear"] >= -0.030
          and metrics["right_clear"] >= -0.030)

    path = _sim_path(Pose(x=start_x, y=start_y, yaw_rad=yaw_rad), motions)

    return PlannedPath(
        result=PlanResult(
            success=True, practical_success=ok, strict_success=ok,
            reason="analytical_success", planner="analytical_v6",
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
    print(f"R = {R:.4f} m   R_MIN_STEER = {WB/math.tan(math.radians(STEER_MAX)):.4f} m")
    print(f"Boundary: start_x = {-R:.4f} m  (x0_aisle = R = {R:.4f} m)")
    print()

    cases = [
        (-1.28, 0.70, "default  (x0_aisle < R → MAXSTEER)"),
        (-1.50, 0.70, "far     (x0_aisle > R → STRAIGHT)"),
    ]
    for sx, sy, label in cases:
        p = plan_from_start(sx, sy, 180.0, "right_only")
        d = result_to_dict(p)
        m = d["metrics"]
        print(f"--- {label} ---")
        print(f"  move1_path : {m['move1_path']}  ({m['move1_reason']})")
        print(f"  d1         : {m['d1_m']:.4f} m  steer1 = {m['steer1_deg']:+.1f}°")
        print(f"  s23        : {m['s23_m']:.4f} m  (arc)")
        print(f"  d4 fallback: {m['d4_m']:.4f} m  (rear_cam overrides)")
        print(f"  est. times : M1={m['est_d1_time_s']:.1f}s  arc={m['est_arc_time_s']:.1f}s  M3={m['est_d4_time_s']:.1f}s")
        total = m['est_d1_time_s'] + m['est_arc_time_s'] + m['est_d4_time_s']
        print(f"  drive total: {total:.1f} s  + overhead → ~{total+7:.0f} s")
        print()
