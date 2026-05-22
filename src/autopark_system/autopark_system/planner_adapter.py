from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import math

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


# ============================================================
# Math helpers
# ============================================================

def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / math.pi


def wrap_to_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_err_deg(yaw: float, target_yaw: float) -> float:
    return abs(rad2deg(wrap_to_pi(yaw - target_yaw)))


# ============================================================
# Data classes
# ============================================================

@dataclass
class Pose:
    x: float
    y: float
    yaw: float


@dataclass
class VehicleSpec:
    length: float = 1.30
    width: float = 0.67
    wheelbase: float = 0.739
    front_overhang: float = 0.355
    rear_overhang: float = 0.170
    max_steer_deg: float = 30.0

    @property
    def half_width(self) -> float:
        return self.width * 0.5

    @property
    def front_extent_from_rear_axle(self) -> float:
        return self.wheelbase + self.front_overhang

    @property
    def rear_extent_from_rear_axle(self) -> float:
        return self.rear_overhang


@dataclass
class RectObstacle:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class ParkingEnv:
    # User real map:
    # origin = middle slot entrance center
    # middle slot: x [-0.38, +0.38], y [-1.39, 0.0]
    slot_width: float = 0.76
    slot_depth: float = 1.39
    left_obstacle: RectObstacle | None = None
    right_obstacle: RectObstacle | None = None

    @property
    def middle_x_min(self) -> float:
        return -self.slot_width * 0.5

    @property
    def middle_x_max(self) -> float:
        return +self.slot_width * 0.5

    @property
    def slot_front_y(self) -> float:
        return 0.0

    @property
    def slot_back_y(self) -> float:
        return -self.slot_depth


@dataclass
class CandidateResult:
    success: bool
    reason: str
    planner: str
    case_name: str
    primitive_seq: List[Tuple[str, float, float]]
    path: List[Pose]
    final_pose: Pose
    metrics: Dict[str, float]
    strict_success: bool
    practical_success: bool


class PlannedPath:
    def __init__(self, result: CandidateResult, motions: List[dict]):
        self.result = result
        self.motions = motions


# ============================================================
# Geometry
# ============================================================

def vehicle_corners(p: Pose, spec: VehicleSpec) -> List[Tuple[float, float]]:
    c = math.cos(p.yaw)
    s = math.sin(p.yaw)

    fx = spec.front_extent_from_rear_axle
    rx = -spec.rear_extent_from_rear_axle
    hw = spec.half_width

    pts_local = [
        (fx, +hw),
        (fx, -hw),
        (rx, -hw),
        (rx, +hw),
    ]

    pts_world = []
    for lx, ly in pts_local:
        wx = p.x + c * lx - s * ly
        wy = p.y + s * lx + c * ly
        pts_world.append((wx, wy))

    return pts_world


def aabb_of_poly(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def aabb_intersects(a, b) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b

    if ax1 < bx0 or bx1 < ax0:
        return False
    if ay1 < by0 or by1 < ay0:
        return False

    return True


def collides_rect(p: Pose, spec: VehicleSpec, rect: RectObstacle) -> bool:
    car_aabb = aabb_of_poly(vehicle_corners(p, spec))
    obs_aabb = (rect.x_min, rect.x_max, rect.y_min, rect.y_max)
    return aabb_intersects(car_aabb, obs_aabb)


def out_of_bounds(p: Pose) -> bool:
    # allow aisle above slot and wall-side inside slot
    return (p.x < -2.20 or p.x > 2.20 or p.y < -1.70 or p.y > 1.40)


def collides_any(p: Pose, spec: VehicleSpec, env: ParkingEnv) -> bool:
    if out_of_bounds(p):
        return True

    if env.left_obstacle is not None and collides_rect(p, spec, env.left_obstacle):
        return True

    if env.right_obstacle is not None and collides_rect(p, spec, env.right_obstacle):
        return True

    return False


def slot_metrics(env: ParkingEnv, p: Pose, spec: VehicleSpec, target_yaw: float) -> Dict[str, float]:
    poly = vehicle_corners(p, spec)
    x0, x1, y0, y1 = aabb_of_poly(poly)

    left_clear = x0 - env.middle_x_min
    right_clear = env.middle_x_max - x1
    front_clear = env.slot_front_y - y1
    rear_clear = y0 - env.slot_back_y

    inside_slot = (
        x0 >= env.middle_x_min and
        x1 <= env.middle_x_max and
        y0 >= env.slot_back_y and
        y1 <= env.slot_front_y
    )

    return {
        "left_clear": left_clear,
        "right_clear": right_clear,
        "front_clear": front_clear,
        "rear_clear": rear_clear,
        "yaw_err_deg": yaw_err_deg(p.yaw, target_yaw),
        "inside_slot": inside_slot,
        "aabb_x_min": x0,
        "aabb_x_max": x1,
        "aabb_y_min": y0,
        "aabb_y_max": y1,
    }


# ============================================================
# Motion simulation
# ============================================================

Primitive = Tuple[str, float, float]  # gear, steer_deg, length_m


def merge_same_primitives(seq: List[Primitive]) -> List[Primitive]:
    if not seq:
        return []

    out = [seq[0]]

    for gear, steer_deg, length in seq[1:]:
        pg, ps, pl = out[-1]
        if gear == pg and abs(steer_deg - ps) < 1e-12:
            out[-1] = (pg, ps, pl + length)
        else:
            out.append((gear, steer_deg, length))

    return out


def maneuver_count(seq: List[Primitive]) -> int:
    if not seq:
        return 0

    count = 0
    prev = seq[0][0]

    for i in range(1, len(seq)):
        if seq[i][0] != prev:
            count += 1
            prev = seq[i][0]

    return count


def step_pose(p: Pose, gear: str, steer_deg: float, ds: float, spec: VehicleSpec) -> Pose:
    steer = deg2rad(steer_deg)
    ds_signed = ds if gear == "f" else -ds

    if abs(steer) < 1e-12:
        return Pose(
            x=p.x + ds_signed * math.cos(p.yaw),
            y=p.y + ds_signed * math.sin(p.yaw),
            yaw=p.yaw,
        )

    kappa = math.tan(steer) / spec.wheelbase
    dtheta = ds_signed * kappa
    radius = 1.0 / kappa

    cx = p.x - radius * math.sin(p.yaw)
    cy = p.y + radius * math.cos(p.yaw)

    nyaw = wrap_to_pi(p.yaw + dtheta)
    nx = cx + radius * math.sin(nyaw)
    ny = cy - radius * math.cos(nyaw)

    return Pose(nx, ny, nyaw)


def simulate_sequence(
    start: Pose,
    seq: List[Primitive],
    spec: VehicleSpec,
    env: ParkingEnv,
    step_len: float = 0.01,
) -> Tuple[bool, List[Pose], Pose]:
    p = start
    path = [start]

    for gear, steer_deg, length in seq:
        n = max(1, int(math.ceil(abs(length) / step_len)))
        ds = abs(length) / n

        for _ in range(n):
            p = step_pose(p, gear, steer_deg, ds, spec)
            path.append(p)

            if collides_any(p, spec, env):
                return False, path, p

    return True, path, p


# ============================================================
# Environment
# ============================================================

def build_env(case_name: str) -> ParkingEnv:
    env = ParkingEnv()

    # Adjacent fake cars beside middle slot.
    # Slot entrance line = y 0.0
    # Wall side = y -1.39
    left_car = RectObstacle(
        x_min=-1.095,
        x_max=-0.425,
        y_min=-1.345,
        y_max=-0.045,
    )

    right_car = RectObstacle(
        x_min=0.425,
        x_max=1.095,
        y_min=-1.345,
        y_max=-0.045,
    )

    if case_name == "left_only":
        env.left_obstacle = left_car
    elif case_name == "right_only":
        env.right_obstacle = right_car
    elif case_name == "both_sides":
        env.left_obstacle = left_car
        env.right_obstacle = right_car
    else:
        raise ValueError("Unknown case_name: " + str(case_name))

    return env


# ============================================================
# Planner
# ============================================================

class ThreeCasePracticalPlanner:
    def __init__(self, env: ParkingEnv, spec: VehicleSpec):
        self.env = env
        self.spec = spec

        # Final parked car:
        # front points to aisle, rear points to wall
        self.target_yaw = deg2rad(90.0)

        self.target_rear_axle_x = 0.0

        # rear axle target that centers body in slot depth
        self.target_rear_axle_y = -(
            self.env.slot_depth
            + self.spec.front_extent_from_rear_axle
            - self.spec.rear_extent_from_rear_axle
        ) * 0.5

    def strict_success(self, p: Pose) -> Tuple[bool, Dict[str, float]]:
        m = slot_metrics(self.env, p, self.spec, self.target_yaw)

        ok = (
            m["inside_slot"]
            and 0.02 <= m["left_clear"] <= 0.07
            and 0.02 <= m["right_clear"] <= 0.07
            and 0.02 <= m["front_clear"] <= 0.07
            and 0.02 <= m["rear_clear"] <= 0.07
            and m["yaw_err_deg"] <= 3.0
        )

        return ok, m

    def practical_success(self, p: Pose) -> Tuple[bool, Dict[str, float]]:
        m = slot_metrics(self.env, p, self.spec, self.target_yaw)

        ok = (
            m["inside_slot"]
            and m["left_clear"] >= 0.0
            and m["right_clear"] >= 0.0
            and m["front_clear"] >= 0.0
            and m["rear_clear"] >= 0.0
            and m["yaw_err_deg"] <= 12.0
        )

        return ok, m

    def final_score(self, p: Pose, seq: List[Primitive]) -> float:
        m = slot_metrics(self.env, p, self.spec, self.target_yaw)

        score = 0.0
        score += 500.0 * abs(p.x - self.target_rear_axle_x)
        score += 2500.0 * abs(p.y - self.target_rear_axle_y)
        score += 15.0 * m["yaw_err_deg"]

        for key in ("left_clear", "right_clear", "front_clear", "rear_clear"):
            if m[key] < 0.0:
                score += 50000.0 * (-m[key])

        if not m["inside_slot"]:
            score += 1500.0

        score += 5.0 * len(seq)
        score += 25.0 * maneuver_count(seq)

        return score

    def generate_center_templates(self) -> List[List[Primitive]]:
        templates: List[List[Primitive]] = []

        # Real-layout template for start:
        # x around 0, y around +0.70, yaw 180 deg, car comes from right to left.
        #
        # Phase 1: short forward-left pre-position
        # Phase 2: reverse arc into middle slot
        # Phase 3: reverse straighten / depth correction

        for pre_steer in (-18.0, -15.0, -12.0):
            for pre_len in (0.90, 1.00, 1.10, 1.20):
                for r1_steer in (18.0, 20.0, 22.0, 25.0):
                    for r1_len in (1.60, 1.80, 2.00):
                        for r2_steer in (0.0, 8.0, 10.0, 12.0):
                            for r2_len in (0.30, 0.45, 0.60):
                                seq = [
                                    ("f", pre_steer, pre_len),
                                    ("r", r1_steer, r1_len),
                                    ("r", r2_steer, r2_len),
                                ]
                                templates.append(merge_same_primitives(seq))

        # Slightly shorter starts
        for pre_len in (0.60, 0.75, 0.90):
            for r1_steer in (22.0, 25.0, 28.0, 30.0):
                for r1_len in (1.30, 1.50, 1.70):
                    for r2_len in (0.30, 0.50, 0.70):
                        seq = [
                            ("f", -15.0, pre_len),
                            ("r", r1_steer, r1_len),
                            ("r", 8.0, r2_len),
                        ]
                        templates.append(merge_same_primitives(seq))

        # If start is already offset, allow almost no pre-position
        for pre_len in (0.00, 0.20, 0.40):
            for r1_steer in (25.0, 28.0, 30.0):
                for r1_len in (1.10, 1.30, 1.50):
                    for r2_steer in (0.0, 8.0, 12.0):
                        for r2_len in (0.20, 0.40, 0.60):
                            seq = []
                            if pre_len > 0.0:
                                seq.append(("f", -12.0, pre_len))
                            seq += [
                                ("r", r1_steer, r1_len),
                                ("r", r2_steer, r2_len),
                            ]
                            templates.append(merge_same_primitives(seq))

        return templates

    def generate_templates(self, case_name: str) -> List[List[Primitive]]:
        # All 3 cases park into the middle slot.
        # The case only changes collision obstacles.
        return self.generate_center_templates()

    def plan(self, case_name: str, start: Pose, debug: bool = False) -> CandidateResult:
        templates = self.generate_templates(case_name)

        best_success = None
        best_success_score = float("inf")

        best_failed_seq: List[Primitive] = []
        best_failed_path: List[Pose] = [start]
        best_failed_pose = start
        best_failed_metrics = slot_metrics(self.env, start, self.spec, self.target_yaw)
        best_failed_score = float("inf")

        for seq in templates:
            ok, path, end_pose = simulate_sequence(start, seq, self.spec, self.env, step_len=0.01)

            if not ok:
                continue

            sc = self.final_score(end_pose, seq)

            strict_ok, strict_m = self.strict_success(end_pose)
            if strict_ok and sc < best_success_score:
                best_success_score = sc
                best_success = CandidateResult(
                    success=True,
                    reason="strict_success",
                    planner="real_middle_slot_planner",
                    case_name=case_name,
                    primitive_seq=seq,
                    path=path,
                    final_pose=end_pose,
                    metrics=strict_m,
                    strict_success=True,
                    practical_success=True,
                )

            if best_success is None:
                practical_ok, practical_m = self.practical_success(end_pose)
                if practical_ok and sc < best_success_score:
                    best_success_score = sc
                    best_success = CandidateResult(
                        success=True,
                        reason="practical_success",
                        planner="real_middle_slot_planner",
                        case_name=case_name,
                        primitive_seq=seq,
                        path=path,
                        final_pose=end_pose,
                        metrics=practical_m,
                        strict_success=False,
                        practical_success=True,
                    )

            if sc < best_failed_score:
                best_failed_score = sc
                best_failed_seq = seq
                best_failed_path = path
                best_failed_pose = end_pose
                best_failed_metrics = slot_metrics(self.env, end_pose, self.spec, self.target_yaw)

        if best_success is not None:
            return best_success

        return CandidateResult(
            success=False,
            reason="best_failed",
            planner="real_middle_slot_planner(best_failed)",
            case_name=case_name,
            primitive_seq=best_failed_seq,
            path=best_failed_path,
            final_pose=best_failed_pose,
            metrics=best_failed_metrics,
            strict_success=False,
            practical_success=False,
        )


# ============================================================
# ROS adapter API
# ============================================================

def plan_from_start(
    start_x: float,
    start_y: float,
    start_yaw_deg: float,
    planner_mode: str = "both_sides",
) -> PlannedPath:
    case_name = planner_mode.strip().lower()

    if case_name not in ("left_only", "right_only", "both_sides"):
        case_name = "both_sides"

    env = build_env(case_name)
    spec = VehicleSpec()

    start = Pose(start_x, start_y, deg2rad(start_yaw_deg))

    planner = ThreeCasePracticalPlanner(env, spec)
    result = planner.plan(case_name, start, debug=False)

    motions = [
        {
            "gear": 1 if gear == "f" else -1,
            "steer_deg": float(steer_deg),
            "dist_m": float(dist_m),
        }
        for gear, steer_deg, dist_m in result.primitive_seq
    ]

    # IMPORTANT:
    # Do not use tiny fallback like 0.06 / 0.05 m.
    # If planner result has empty primitive_seq, use a real practical
    # template distance for real-car testing.
    if len(motions) == 0:
        result.reason = "empty_planner_result_using_realcar_fallback_template"
        result.planner = "real_middle_slot_planner(realcar_fallback_template)"
        result.primitive_seq = [
            ("f", -15.0, 0.60),
            ("r", 25.0, 0.90),
            ("r", 8.0, 0.35),
        ]

        ok, path, end_pose = simulate_sequence(
            start,
            result.primitive_seq,
            spec,
            env,
            step_len=0.01,
        )

        result.path = path
        result.final_pose = end_pose
        result.metrics = slot_metrics(env, end_pose, spec, planner.target_yaw)

        motions = [
            {
                "gear": 1 if gear == "f" else -1,
                "steer_deg": float(steer_deg),
                "dist_m": float(dist_m),
            }
            for gear, steer_deg, dist_m in result.primitive_seq
        ]

    return PlannedPath(result, motions)

def result_to_dict(planned: PlannedPath) -> Dict[str, Any]:
    res = planned.result
    metrics = dict(res.metrics) if isinstance(res.metrics, dict) else {}

    return {
        "success": res.success,
        "reason": res.reason,
        "planner": res.planner,
        "case_name": res.case_name,
        "maneuvers": maneuver_count(res.primitive_seq),
        "motions": planned.motions,
        "path": [
            {
                "x": p.x,
                "y": p.y,
                "yaw_rad": p.yaw,
            }
            for p in res.path
        ],
        "final_pose": {
            "x": res.final_pose.x,
            "y": res.final_pose.y,
            "yaw_rad": res.final_pose.yaw,
        },
        "metrics": metrics,
        "left_clear": metrics.get("left_clear"),
        "right_clear": metrics.get("right_clear"),
        "front_clear": metrics.get("front_clear"),
        "rear_clear": metrics.get("rear_clear"),
        "yaw_err_deg": metrics.get("yaw_err_deg"),
        "inside_slot": metrics.get("inside_slot"),
        "practical_success": res.practical_success,
        "strict_success": res.strict_success,
    }


def build_ros_path(path_points, frame_id="map") -> Path:
    msg = Path()
    msg.header.frame_id = frame_id

    for p in path_points:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(p["x"])
        ps.pose.position.y = float(p["y"])
        ps.pose.orientation.z = float(p.get("yaw_rad", 0.0))
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)

    return msg
