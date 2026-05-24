import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


Primitive = Tuple[str, float, float]
# Primitive = (direction, steer_deg, dist_m)
# direction: "f" = forward, "r" = reverse


@dataclass
class Pose:
    x: float
    y: float
    yaw_rad: float


@dataclass
class VehicleSpec:
    # Real car geometry, pose is rear axle center.
    width: float = 0.67
    wheelbase: float = 0.739
    front_overhang: float = 0.355
    rear_overhang: float = 0.170
    max_steer_deg: float = 22.0

    @property
    def half_width(self) -> float:
        return self.width * 0.5

    @property
    def front_extent(self) -> float:
        # rear axle center -> front bumper
        return self.wheelbase + self.front_overhang

    @property
    def rear_extent(self) -> float:
        # rear axle center -> rear bumper
        return self.rear_overhang


@dataclass
class RectObstacle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class ParkingEnv:
    # Coordinate convention:
    # slot entrance/mouth center = (0, 0)
    # slot extends backward to y = -slot_depth
    slot_width: float = 0.76
    slot_depth: float = 1.39

    # Real safety.
    hard_wall_clearance_m: float = 0.035
    target_rear_clearance_m: float = 0.055

    # Allow tiny line/tape error for practical test.
    side_clearance_allowance_m: float = -0.18
    front_clearance_allowance_m: float = -0.15

    # Obstacles for side cases. They are optional because your real slot
    # currently uses tape/wall more than box obstacles.
    left_obstacle: Optional[RectObstacle] = None
    right_obstacle: Optional[RectObstacle] = None

    @property
    def half_slot_width(self) -> float:
        return self.slot_width * 0.5

    @property
    def slot_back_y(self) -> float:
        return -self.slot_depth

    @property
    def slot_front_y(self) -> float:
        return 0.0


@dataclass
class CandidateResult:
    success: bool
    reason: str
    planner: str
    case_name: str
    primitive_seq: List[Primitive]
    path: List[Pose]
    metrics: Dict[str, float]
    practical_success: bool = False
    strict_success: bool = False
    score: float = 1e18


@dataclass
class PlannedPath:
    result: CandidateResult
    motions: List[Dict[str, Any]]


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def angle_err_deg(a: float, b: float) -> float:
    return abs(math.degrees(wrap_pi(a - b)))


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def vehicle_corners(p: Pose, spec: VehicleSpec) -> List[Tuple[float, float]]:
    """
    Vehicle rectangle corners from rear axle center pose.
    x local = forward direction from rear axle
    y local = left side direction
    """
    front = spec.front_extent
    rear = -spec.rear_extent
    hw = spec.half_width

    local = [
        (front, hw),
        (front, -hw),
        (rear, -hw),
        (rear, hw),
    ]

    c = math.cos(p.yaw_rad)
    s = math.sin(p.yaw_rad)

    pts = []
    for lx, ly in local:
        wx = p.x + lx * c - ly * s
        wy = p.y + lx * s + ly * c
        pts.append((wx, wy))
    return pts


def aabb_of_poly(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [q[0] for q in poly]
    ys = [q[1] for q in poly]
    return min(xs), max(xs), min(ys), max(ys)


def rect_intersects_aabb(a: Tuple[float, float, float, float], r: RectObstacle) -> bool:
    xmin, xmax, ymin, ymax = a
    if xmax < r.xmin:
        return False
    if xmin > r.xmax:
        return False
    if ymax < r.ymin:
        return False
    if ymin > r.ymax:
        return False
    return True


def out_of_bounds(p: Pose) -> bool:
    # Keep search finite but generous.
    if p.x < -3.0 or p.x > 3.0:
        return True
    if p.y < -2.2 or p.y > 2.5:
        return True
    return False


def collides_rect(p: Pose, spec: VehicleSpec, obs: RectObstacle) -> bool:
    aabb = aabb_of_poly(vehicle_corners(p, spec))
    return rect_intersects_aabb(aabb, obs)

def collides_any(p: Pose, spec: VehicleSpec, env: ParkingEnv) -> bool:
    if out_of_bounds(p):
        return True

    car_aabb = aabb_of_poly(vehicle_corners(p, spec))
    _, car_x_max, _, _ = car_aabb

    # Back wall safety.
    # Wall is x = slot_depth.
    # Rear bumper / body must not pass too close to wall.
    wall_safe_x = env.slot_depth - env.hard_wall_clearance_m
    if car_x_max > wall_safe_x:
        return True

    if env.left_obstacle is not None and collides_rect(p, spec, env.left_obstacle):
        return True

    if env.right_obstacle is not None and collides_rect(p, spec, env.right_obstacle):
        return True

    return False

def step_bicycle(p: Pose, gear: int, steer_deg: float, ds: float, spec: VehicleSpec) -> Pose:
    """
    Kinematic bicycle step at rear axle center.
    gear: +1 forward, -1 reverse
    ds: positive distance step
    """
    steer_deg = clamp(steer_deg, -spec.max_steer_deg, spec.max_steer_deg)
    delta = math.radians(steer_deg)

    direction = 1.0 if gear >= 0 else -1.0
    signed_ds = direction * ds

    if abs(delta) < math.radians(0.2):
        x = p.x + signed_ds * math.cos(p.yaw_rad)
        y = p.y + signed_ds * math.sin(p.yaw_rad)
        yaw = p.yaw_rad
        return Pose(x, y, wrap_pi(yaw))

    yaw_rate_per_m = math.tan(delta) / spec.wheelbase
    dyaw = signed_ds * yaw_rate_per_m

    # Small-step integration is enough because caller uses small ds.
    mid_yaw = p.yaw_rad + 0.5 * dyaw
    x = p.x + signed_ds * math.cos(mid_yaw)
    y = p.y + signed_ds * math.sin(mid_yaw)
    yaw = wrap_pi(p.yaw_rad + dyaw)
    return Pose(x, y, yaw)


def simulate_sequence(
    start: Pose,
    seq: List[Primitive],
    spec: VehicleSpec,
    env: ParkingEnv,
    step_m: float = 0.04,
) -> Tuple[List[Pose], bool]:
    path = [start]
    p = start
    collided = False

    for direction, steer_deg, dist_m in seq:
        gear = 1 if direction == "f" else -1
        remaining = abs(float(dist_m))

        while remaining > 1e-9:
            ds = min(step_m, remaining)
            p = step_bicycle(p, gear, steer_deg, ds, spec)
            path.append(p)

            if collides_any(p, spec, env):
                collided = True
                return path, collided

            remaining -= ds

    return path, collided


def merge_same_primitives(seq: List[Primitive]) -> List[Primitive]:
    merged: List[Primitive] = []

    for direction, steer, dist in seq:
        if abs(dist) < 1e-6:
            continue

        if not merged:
            merged.append((direction, steer, dist))
            continue

        pd, ps, pl = merged[-1]
        if pd == direction and abs(ps - steer) < 1e-6:
            merged[-1] = (pd, ps, pl + dist)
        else:
            merged.append((direction, steer, dist))

    return merged


def maneuver_count(seq: List[Primitive]) -> int:
    if not seq:
        return 0

    count = 1
    last = seq[0][0]
    for d, _, _ in seq[1:]:
        if d != last:
            count += 1
            last = d
    return count

def slot_metrics(env: ParkingEnv, p: Pose, spec: VehicleSpec, target_yaw: float) -> Dict[str, float]:
    poly = vehicle_corners(p, spec)
    xmin, xmax, ymin, ymax = aabb_of_poly(poly)

    # New convention:
    # slot entrance/front = x = 0
    # slot back wall = x = +slot_depth
    # slot width is along y

    left_clear = env.half_slot_width - ymax
    right_clear = ymin + env.half_slot_width

    # Rear bumper is the largest x side when final yaw is about 180 deg.
    rear_clear = env.slot_depth - xmax

    # Front bumper must be inside the mouth, x >= 0.
    front_clear = xmin - 0.0

    yaw_err = angle_err_deg(p.yaw_rad, target_yaw)

    inside_slot = (
        left_clear >= 0.0
        and right_clear >= 0.0
        and rear_clear >= 0.0
        and front_clear >= 0.0
    )

    practical_inside_slot = (
        left_clear >= env.side_clearance_allowance_m
        and right_clear >= env.side_clearance_allowance_m
        and rear_clear >= env.hard_wall_clearance_m
        and front_clear >= env.front_clearance_allowance_m
    )

    return {
        "x": p.x,
        "y": p.y,
        "yaw_rad": p.yaw_rad,
        "yaw_deg": math.degrees(p.yaw_rad),
        "yaw_err_deg": yaw_err,
        "aabb_x_min": xmin,
        "aabb_x_max": xmax,
        "aabb_y_min": ymin,
        "aabb_y_max": ymax,
        "left_clear": left_clear,
        "right_clear": right_clear,
        "front_clear": front_clear,
        "rear_clear": rear_clear,
        "inside_slot": 1.0 if inside_slot else 0.0,
        "practical_inside_slot": 1.0 if practical_inside_slot else 0.0,
    }

class RealMiddleSlotPlanner:
    """
    Flexible practical planner for the real small car.

    It searches many primitive sequences from the current start pose.
    It does not return a single fixed fallback path.

    Pose convention:
      pose = rear axle center
    """

    def __init__(self, spec: VehicleSpec, env: ParkingEnv, case_name: str):
        self.spec = spec
        self.env = env
        self.case_name = case_name

        # Final target:
        # rear axle y should be wall + rear_overhang + rear clearance.
        self.target_rear_axle_x = (
            self.env.slot_depth
            - self.spec.rear_overhang
            - self.env.target_rear_clearance_m
        )
        self.target_rear_axle_y = 0.0

        # Final car should face same yaw as backing-in pose.
        self.target_yaw = math.radians(180.0)

    def plan(self, start: Pose) -> CandidateResult:
        templates = self.generate_templates()

        best: Optional[CandidateResult] = None
        best_success: Optional[CandidateResult] = None

        for seq in templates:
            path, collided = simulate_sequence(start, seq, self.spec, self.env)

            final_pose = path[-1]
            metrics = slot_metrics(self.env, final_pose, self.spec, self.target_yaw)

            strict_success = self.is_strict_success(metrics, collided)
            practical_success = self.is_practical_success(metrics, collided)
            success = strict_success or practical_success

            score = self.score_candidate(metrics, seq, collided)

            if collided:
                reason = "collision"
            elif strict_success:
                reason = "strict_success"
            elif practical_success:
                reason = "practical_success"
            else:
                reason = "best_failed"

            cand = CandidateResult(
                success=success,
                reason=reason,
                planner="real_middle_slot_planner",
                case_name=self.case_name,
                primitive_seq=seq,
                path=path,
                metrics=metrics,
                practical_success=practical_success,
                strict_success=strict_success,
                score=score,
            )

            if best is None or cand.score < best.score:
                best = cand

            if success:
                if best_success is None or cand.score < best_success.score:
                    best_success = cand

        if best_success is not None:
            return best_success

        if best is None:
            return CandidateResult(
                success=False,
                reason="no_candidate",
                planner="real_middle_slot_planner(no_candidate)",
                case_name=self.case_name,
                primitive_seq=[],
                path=[start],
                metrics={},
                score=1e18,
            )

        best.success = False
        best.reason = "best_failed"
        best.planner = "real_middle_slot_planner(best_failed)"
        return best

    def is_strict_success(self, m: Dict[str, float], collided: bool) -> bool:
        if collided:
            return False

        return (
            bool(m.get("inside_slot", 0.0) > 0.5)
            and m["yaw_err_deg"] <= 15.0
            and 0.020 <= m["rear_clear"] <= 0.090
            and m["left_clear"] >= 0.010
            and m["right_clear"] >= 0.010
            and m["front_clear"] >= 0.000
        )

    def is_practical_success(self, m: Dict[str, float], collided: bool) -> bool:
        if collided:
            return False

        # Practical first-pass real-car target.
        # This lets planner find a safe path before strict 3 deg / 45 mm tuning.
        return (
            bool(m.get("practical_inside_slot", 0.0) > 0.5)
            and m["yaw_err_deg"] <= 35.0
            and 0.035 <= m["rear_clear"] <= 0.180
            and m["left_clear"] >= self.env.side_clearance_allowance_m
            and m["right_clear"] >= self.env.side_clearance_allowance_m
            and m["front_clear"] >= self.env.front_clearance_allowance_m
        )

    def score_candidate(self, m: Dict[str, float], seq: List[Primitive], collided: bool) -> float:
        if not m:
            return 1e18

        score = 0.0

        if collided:
            score += 1e9

        # Target rear axle placement.
        score += 700.0 * abs(m["x"] - self.target_rear_axle_x)
        score += 900.0 * abs(m["y"] - self.target_rear_axle_y)

        # Target yaw.
        score += 35.0 * m["yaw_err_deg"]

        # Rear clearance should be around target, not near wall.
        rear_err = abs(m["rear_clear"] - self.env.target_rear_clearance_m)
        score += 2500.0 * rear_err

        # Penalize wall too close hard.
        if m["rear_clear"] < self.env.hard_wall_clearance_m:
            score += 200000.0 * (self.env.hard_wall_clearance_m - m["rear_clear"])

        # Penalize outside slot.
        if m["left_clear"] < 0.0:
            score += 9000.0 * abs(m["left_clear"])
        if m["right_clear"] < 0.0:
            score += 9000.0 * abs(m["right_clear"])
        if m["front_clear"] < 0.0:
            score += 6500.0 * abs(m["front_clear"])

        # Side balance.
        score += 800.0 * abs(m["left_clear"] - m["right_clear"])

        # Prefer successful path strongly.
        if self.is_strict_success(m, collided):
            score -= 200000.0
        elif self.is_practical_success(m, collided):
            score -= 100000.0

        # Prefer shorter and fewer direction changes, but not too strongly.
        total_len = sum(abs(x[2]) for x in seq)
        score += 45.0 * total_len
        score += 15.0 * len(seq)
        score += 25.0 * maneuver_count(seq)

        return score

    def generate_templates(self) -> List[List[Primitive]]:
        base = self.generate_right_only_templates()
        mirrored = self.mirror_templates(base)

        # Car comes from RIGHT (y > 0): use base templates only.
        # Base has r1_steer = +22° which swings rear LEFT toward slot centre.
        # Mirrored has r1_steer = −22° which spirals car away — WRONG for right side.
        if self.case_name == "right_only":
            return base

        # Car comes from LEFT (y < 0): use mirrored only.
        if self.case_name == "left_only":
            return mirrored

        # Both sides: try all templates.
        return base + mirrored

    def mirror_templates(self, templates: List[List[Primitive]]) -> List[List[Primitive]]:
        mirrored: List[List[Primitive]] = []
        for seq in templates:
            new_seq = []
            for d, steer, dist in seq:
                new_seq.append((d, -steer, dist))
            mirrored.append(new_seq)
        return mirrored

    def generate_right_only_templates(self) -> List[List[Primitive]]:
        templates: List[List[Primitive]] = []

        # Fast focused search after coordinate fix.
        # Goal:
        #   x ≈ 1.05–1.16
        #   y ≈ 0
        #   yaw ≈ 180
        #
        # Important:
        # r1 turns into slot.
        # r2 counter-steers to straighten yaw.
        # r3 small straight insert.

        # 3-move focused candidates
        for pre_steer in (-22.0, -20.0, -18.0):
            for pre_len in (0.25, 0.35, 0.45):
                for r1_steer in (22.0, 20.0):
                    for r1_len in (0.50, 0.60, 0.70, 0.80, 0.90):
                        for r2_steer in (-22.0, -20.0, -18.0):
                            for r2_len in (0.75, 0.85, 0.95, 1.05):
                                seq = [
                                    ("f", pre_steer, pre_len),
                                    ("r", r1_steer, r1_len),
                                    ("r", r2_steer, r2_len),
                                ]
                                templates.append(merge_same_primitives(seq))

        # 4-move: add short straight after yaw correction
        for pre_steer in (-22.0, -20.0, -18.0):
            for pre_len in (0.25, 0.35, 0.45):
                for r1_steer in (22.0, 20.0):
                    for r1_len in (0.55, 0.65, 0.75, 0.85):
                        for r2_steer in (-22.0, -20.0, -18.0):
                            for r2_len in (0.75, 0.85, 0.95):
                                for r3_len in (0.00, 0.05, 0.10, 0.15):
                                    seq = [
                                        ("f", pre_steer, pre_len),
                                        ("r", r1_steer, r1_len),
                                        ("r", r2_steer, r2_len),
                                        ("r", 0.0, r3_len),
                                    ]
                                    templates.append(merge_same_primitives(seq))

        # 5-move correction: if 3 reverse parts cannot center/yaw enough
        for pre_steer in (-22.0, -20.0):
            for pre_len in (0.25, 0.35):
                for r1_steer in (22.0, 20.0):
                    for r1_len in (0.55, 0.70):
                        for f2_steer in (8.0, 12.0):
                            for f2_len in (0.10, 0.20):
                                for r2_steer in (-22.0, -20.0):
                                    for r2_len in (0.55, 0.70):
                                        for r3_len in (0.05, 0.15):
                                            seq = [
                                                ("f", pre_steer, pre_len),
                                                ("r", r1_steer, r1_len),
                                                ("f", f2_steer, f2_len),
                                                ("r", r2_steer, r2_len),
                                                ("r", 0.0, r3_len),
                                            ]
                                            templates.append(merge_same_primitives(seq))

        return templates

def make_env(case_name: str) -> ParkingEnv:
    env = ParkingEnv()

    # Fake car (67×130 cm) centred in adjacent 76 cm slot.
    # Middle slot: y ∈ [−0.38, +0.38]
    # Right slot fake car inner edge at y ≈ +0.50 (with gap)
    # Left  slot fake car inner edge at y ≈ −0.50 (with gap)
    # This prevents the planner from choosing spiral paths that cross slot lines.

    if case_name in ("right_only", "both_sides"):
        env.right_obstacle = RectObstacle(
            xmin=0.020, xmax=1.370,
            ymin=0.500, ymax=1.095,
        )

    if case_name in ("left_only", "both_sides"):
        env.left_obstacle = RectObstacle(
            xmin=0.020, xmax=1.370,
            ymin=-1.095, ymax=-0.500,
        )

    return env

def primitives_to_motions(seq: List[Primitive]) -> List[Dict[str, Any]]:
    motions: List[Dict[str, Any]] = []

    for direction, steer_deg, dist_m in seq:
        gear = 1 if direction == "f" else -1
        motions.append(
            {
                "gear": gear,
                "steer_deg": float(steer_deg),
                "dist_m": float(abs(dist_m)),
            }
        )

    return motions


def pose_to_dict(p: Pose) -> Dict[str, float]:
    return {
        "x": p.x,
        "y": p.y,
        "yaw_rad": p.yaw_rad,
        "yaw_deg": math.degrees(p.yaw_rad),
    }


def result_to_dict(planned: PlannedPath) -> Dict[str, Any]:
    res = planned.result

    return {
        "success": bool(res.success),
        "ok": bool(res.success),
        "reason": res.reason,
        "planner": res.planner,
        "case_name": res.case_name,
        "maneuvers": maneuver_count(res.primitive_seq),
        "motions": planned.motions,
        "primitive_seq": [
            {
                "direction": d,
                "steer_deg": steer,
                "dist_m": dist,
            }
            for d, steer, dist in res.primitive_seq
        ],
        "path": [pose_to_dict(p) for p in res.path],
        "metrics": res.metrics,
        "score": res.score,
        "practical_success": bool(res.practical_success),
        "strict_success": bool(res.strict_success),
    }


def plan_from_start(
    start_x: float,
    start_y: float,
    start_yaw_deg: float,
    case_name: str = "right_only",
) -> PlannedPath:
    case_name = str(case_name).strip().lower()

    if case_name not in ("left_only", "right_only", "both_sides"):
        case_name = "right_only"

    spec = VehicleSpec()
    env = make_env(case_name)

    start = Pose(
        x=float(start_x),
        y=float(start_y),
        yaw_rad=math.radians(float(start_yaw_deg)),
    )

    planner = RealMiddleSlotPlanner(spec=spec, env=env, case_name=case_name)
    result = planner.plan(start)
    motions = primitives_to_motions(result.primitive_seq)

    return PlannedPath(result=result, motions=motions)


# Optional manual test:
if __name__ == "__main__":
    import time

    t0 = time.time()
    print("planner test started...")

    planned = plan_from_start(0.0, 0.7, 180.0, "right_only")
    d = result_to_dict(planned)
    m = d.get("metrics", {})

    print("planner test finished in", round(time.time() - t0, 2), "sec")
    print("success:", d["success"])
    print("reason:", d["reason"])
    print("planner:", d["planner"])
    print("motions:", d["motions"])
    print("final x,y,yaw:",
          round(m.get("x", 999), 3),
          round(m.get("y", 999), 3),
          round(m.get("yaw_deg", 999), 1))
    print("clear:",
          "L", round(m.get("left_clear", 999), 3),
          "R", round(m.get("right_clear", 999), 3),
          "F", round(m.get("front_clear", 999), 3),
          "Rear", round(m.get("rear_clear", 999), 3),
          "yaw_err", round(m.get("yaw_err_deg", 999), 1))
    print("inside_slot:", m.get("inside_slot"))
    print("practical_inside_slot:", m.get("practical_inside_slot"))
    print("score:", d["score"])
