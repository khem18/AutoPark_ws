#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone BOTH_SIDES planner v11
- side-biased angled pre-entry
- reverse-only deep-entry
- clean standalone test harness

This version fixes the main v11 issue:
pre-entry was rejected too early by an over-strict geometry gate.

Key changes:
- start-side-aware pre-entry acceptance
- side-biased pre-entry scoring
- soft fallback into deep-entry when hard pre-entry gate is empty
- reverse-only deep-entry library with simple rotation sequences

Notes:
- This is a standalone geometric planner/debugger, not ROS code
- Units: meters, radians
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
# Basic math
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


# ============================================================
# Core data structures
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
    # Common frame:
    # x = lateral across slots
    # y = from wall toward aisle
    #
    # Middle slot centered at x=0
    # Slot width = 0.76, depth = 1.39
    slot_width: float = 0.76
    slot_depth: float = 1.39
    lane_y: float = 2.17

    # two parked fake cars on both sides of the middle slot
    # left slot = [-1.14,-0.38], middle = [-0.38,0.38], right = [0.38,1.14]
    left_obstacle: Optional[RectObstacle] = None
    right_obstacle: Optional[RectObstacle] = None

    @property
    def middle_x_min(self) -> float:
        return -self.slot_width * 0.5

    @property
    def middle_x_max(self) -> float:
        return +self.slot_width * 0.5

    @property
    def slot_y_min(self) -> float:
        return 0.0

    @property
    def slot_y_max(self) -> float:
        return self.slot_depth


@dataclass
class PlannerResult:
    success: bool
    reason: str
    planner: str
    maneuvers: int
    path: List[Pose]
    final_pose: Pose
    metrics: dict


# ============================================================
# Vehicle geometry / collision helpers
# ============================================================

def vehicle_corners(p: Pose, spec: VehicleSpec) -> List[Tuple[float, float]]:
    """
    Vehicle rectangle corners using rear axle center as reference.
    Order:
    front-left, front-right, rear-right, rear-left
    """
    cf = math.cos(p.yaw)
    sf = math.sin(p.yaw)

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
        wx = p.x + cf * lx - sf * ly
        wy = p.y + sf * lx + cf * ly
        pts_world.append((wx, wy))
    return pts_world


def aabb_of_polygon(poly: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def aabb_intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    if ax1 < bx0 or bx1 < ax0:
        return False
    if ay1 < by0 or by1 < ay0:
        return False
    return True


def collides_with_obstacle(p: Pose, spec: VehicleSpec, obs: RectObstacle) -> bool:
    car_poly = vehicle_corners(p, spec)
    car_aabb = aabb_of_polygon(car_poly)
    obs_aabb = (obs.x_min, obs.x_max, obs.y_min, obs.y_max)
    return aabb_intersects(car_aabb, obs_aabb)


def collides_any(p: Pose, spec: VehicleSpec, env: ParkingEnv) -> bool:
    if env.left_obstacle is not None and collides_with_obstacle(p, spec, env.left_obstacle):
        return True
    if env.right_obstacle is not None and collides_with_obstacle(p, spec, env.right_obstacle):
        return True
    return False


def middle_slot_metrics(env: ParkingEnv, p: Pose, spec: VehicleSpec) -> dict:
    """
    Approximate slot clearances using vehicle AABB against target middle slot.
    Good enough for planner ranking/debug.
    """
    poly = vehicle_corners(p, spec)
    x0, x1, y0, y1 = aabb_of_polygon(poly)

    left_clear = x0 - env.middle_x_min
    right_clear = env.middle_x_max - x1
    rear_clear = y0 - env.slot_y_min
    front_clear = env.slot_y_max - y1

    yaw_target = deg2rad(-90.0)
    yaw_err_deg = abs(rad2deg(wrap_to_pi(p.yaw - yaw_target)))

    return {
        "left_clear": left_clear,
        "right_clear": right_clear,
        "rear_clear": rear_clear,
        "front_clear": front_clear,
        "yaw_err_deg": yaw_err_deg,
        "inside_slot": (x0 >= env.middle_x_min and x1 <= env.middle_x_max and
                        y0 >= env.slot_y_min and y1 <= env.slot_y_max),
    }


# ============================================================
# Motion primitives
# seq item: (gear, steer_deg, length)
# gear: 'f' or 'r'
# ============================================================

def step_pose(p: Pose, gear: str, steer_deg: float, ds: float, spec: VehicleSpec) -> Pose:
    """
    Bicycle kinematic step around rear axle center.
    ds is positive arc length magnitude; gear decides sign.
    """
    steer = deg2rad(steer_deg)
    direction = +1.0 if gear == 'f' else -1.0
    ds_signed = direction * ds

    if abs(steer) < 1e-9:
        nx = p.x + ds_signed * math.cos(p.yaw)
        ny = p.y + ds_signed * math.sin(p.yaw)
        nyaw = p.yaw
        return Pose(nx, ny, nyaw)

    kappa = math.tan(steer) / spec.wheelbase
    dtheta = ds_signed * kappa
    R = 1.0 / kappa

    cx = p.x - R * math.sin(p.yaw)
    cy = p.y + R * math.cos(p.yaw)
    nyaw = wrap_to_pi(p.yaw + dtheta)
    nx = cx + R * math.sin(nyaw)
    ny = cy - R * math.cos(nyaw)
    return Pose(nx, ny, nyaw)


def simulate_sequence(
    start: Pose,
    seq: List[Tuple[str, float, float]],
    spec: VehicleSpec,
    env: ParkingEnv,
    step_len: float = 0.01
) -> Tuple[bool, List[Pose], Pose]:
    path = [start]
    p = start

    for gear, steer_deg, length in seq:
        n = max(1, int(math.ceil(length / step_len)))
        ds = length / n
        for _ in range(n):
            p = step_pose(p, gear, steer_deg, ds, spec)
            path.append(p)
            if collides_any(p, spec, env):
                return False, path, p

    return True, path, p


def seq_str(seq: List[Tuple[str, float, float]]) -> str:
    if not seq:
        return "[]"
    out = []
    for g, s, l in seq:
        out.append(f"{g}({s:+.1f},{l:.3f})")
    return " -> ".join(out)


def maneuver_count(seq: List[Tuple[str, float, float]]) -> int:
    if not seq:
        return 0
    count = 0
    prev = seq[0][0]
    for i in range(1, len(seq)):
        if seq[i][0] != prev:
            count += 1
            prev = seq[i][0]
    return count


# ============================================================
# BOTH_SIDES planner v11
# ============================================================

class BothSidesPlannerV11:
    def __init__(self, env: ParkingEnv, spec: VehicleSpec):
        self.env = env
        self.spec = spec

    def start_side(self, start: Pose) -> int:
        if start.x < -0.10:
            return -1
        if start.x > 0.10:
            return 1
        return 0

    def pre_entry_sequences(self, start: Pose) -> List[List[Tuple[str, float, float]]]:
        """
        Side-biased angled pre-entry family.
        We allow:
        - straight forward
        - straight reverse bias
        - reverse with one or two small forward alignments
        """
        seqs: List[List[Tuple[str, float, float]]] = []

        # broad families
        reverse_lengths = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        side_angles = [0.0, -20.0, -18.0, -15.0, -12.0, -10.0, -8.0, -6.0,
                       +6.0, +8.0, +10.0, +12.0, +15.0, +18.0, +20.0]

        # simple reverse only
        for s in side_angles:
            for L in reverse_lengths:
                seqs.append([('r', s, L)])

        # reverse + small forward trim
        trims = [0.060, 0.080]
        trim_angles = [+8.0, 0.0, -8.0]
        for s in [-20.0, -18.0, -15.0, -12.0, -10.0, -8.0]:
            for L in [0.10, 0.12, 0.14]:
                for ts in trim_angles:
                    for tL in trims:
                        seqs.append([('r', s, L), ('f', ts, tL)])
        for s in [+8.0, +10.0, +12.0, +15.0, +18.0, +20.0]:
            for L in [0.10, 0.12, 0.14]:
                for ts in trim_angles:
                    for tL in trims:
                        seqs.append([('r', s, L), ('f', ts, tL)])

        return seqs

    def pre_entry_accept(self, start: Pose, p: Pose) -> bool:
        m = middle_slot_metrics(self.env, p, self.spec)
        side = self.start_side(start)

        # near slot mouth / lane-side band
        if not (2.03 <= p.y <= 2.22):
            return False

        # side-biased mouth corridor
        if side < 0:
            if not (-0.42 <= p.x <= -0.06):
                return False
        elif side > 0:
            if not (0.06 <= p.x <= 0.42):
                return False
        else:
            if not (-0.10 <= p.x <= 0.10):
                return False

        yaw_deg = rad2deg(p.yaw)
        d180 = min(abs(yaw_deg - 180.0), abs(yaw_deg + 180.0))

        # committed but not over-rotated
        if not (2.5 <= d180 <= 18.0):
            return False

        # not already vertical
        d_target = abs(rad2deg(wrap_to_pi(p.yaw - deg2rad(-90.0))))
        if d_target < 18.0:
            return False

        # keep physically reasonable
        if m["left_clear"] < -0.10 or m["right_clear"] < -0.10:
            return False

        return True

    def pre_entry_score(self, start: Pose, p: Pose) -> float:
        side = self.start_side(start)

        if side < 0:
            target_x = -0.28
            target_d180 = 8.0
        elif side > 0:
            target_x = +0.28
            target_d180 = 8.0
        else:
            target_x = 0.0
            target_d180 = 4.0

        target_y = 2.09

        yaw_deg = rad2deg(p.yaw)
        d180 = min(abs(yaw_deg - 180.0), abs(yaw_deg + 180.0))

        score = 0.0
        score += 18.0 * abs(p.x - target_x)
        score += 12.0 * abs(p.y - target_y)
        score += 4.0 * abs(d180 - target_d180)
        return score

    def deep_entry_library(self, pre: Pose) -> List[List[Tuple[str, float, float]]]:
        """
        Reverse-only deep-entry rotation library.
        Long-first, then rotating trims.
        """
        seqs: List[List[Tuple[str, float, float]]] = []

        primary_lengths = [0.120, 0.100]
        first_angles = [+30.0, +26.0, +22.0, +18.0, +14.0, +10.0, +6.0,
                        -6.0, -10.0, -14.0, -18.0, -22.0, -26.0, -30.0]

        second_lengths = [0.080, 0.060]
        second_angles = [+18.0, +14.0, +10.0, +6.0, -6.0, -10.0, -14.0, -18.0]

        third_lengths = [0.060]
        third_angles = [+10.0, +6.0, -6.0, -10.0]

        # long-first
        for a1 in first_angles:
            for l1 in primary_lengths:
                seqs.append([('r', a1, l1)])

        # long + rotation trim
        for a1 in first_angles:
            for l1 in primary_lengths:
                for a2 in second_angles:
                    for l2 in second_lengths:
                        seqs.append([('r', a1, l1), ('r', a2, l2)])

        # long + rotation + settle
        for a1 in first_angles:
            for l1 in primary_lengths:
                for a2 in second_angles:
                    for l2 in second_lengths:
                        for a3 in third_angles:
                            for l3 in third_lengths:
                                seqs.append([('r', a1, l1), ('r', a2, l2), ('r', a3, l3)])

        return seqs

    def deep_entry_score(self, p: Pose) -> float:
        m = middle_slot_metrics(self.env, p, self.spec)
        score = 0.0

        # target inside slot, near centered, near vertical
        score += 100.0 * max(0.0, -m["left_clear"])
        score += 100.0 * max(0.0, -m["right_clear"])
        score += 100.0 * max(0.0, -m["rear_clear"])
        score += 100.0 * max(0.0, -m["front_clear"])

        # prefer centered
        score += 8.0 * abs(m["left_clear"] - m["right_clear"])
        # prefer rear inserted but not out through wall
        score += 4.0 * abs(m["rear_clear"] - 0.045)
        # prefer yaw near -90
        score += 2.0 * m["yaw_err_deg"]

        return score

    def validate_final(self, p: Pose) -> Tuple[bool, bool, dict]:
        m = middle_slot_metrics(self.env, p, self.spec)

        strict = (
            0.02 <= m["left_clear"] <= 0.07 and
            0.02 <= m["right_clear"] <= 0.07 and
            0.02 <= m["rear_clear"] <= 0.07 and
            m["yaw_err_deg"] <= 3.0 and
            m["inside_slot"]
        )

        practical = (
            m["inside_slot"] and
            m["yaw_err_deg"] <= 10.0 and
            m["rear_clear"] >= 0.0
        )

        return strict, practical, m

    def plan(self, start: Pose, debug: bool = True) -> PlannerResult:
        pre_entries: List[Tuple[float, Pose, List[Pose], int, list]] = []
        soft_candidates: List[Tuple[float, Pose, List[Pose], int, list]] = []

        if debug:
            print(f"START: x={start.x:.3f}, y={start.y:.3f}, yaw={rad2deg(start.yaw):.3f} deg")
            print("-- PRE-ENTRY DEBUG V11 --")

        seqs = self.pre_entry_sequences(start)

        for i, seq in enumerate(seqs, start=1):
            ok, path, pend = simulate_sequence(start, seq, self.spec, self.env)
            if not ok:
                if debug:
                    print(f"[{i:03d}] reject collision seq={seq_str(seq)}")
                continue

            mans = maneuver_count(seq)
            s = self.pre_entry_score(start, pend)
            soft_candidates.append((s, pend, path, mans, seq))

            if not self.pre_entry_accept(start, pend):
                if debug:
                    print(f"[{i:03d}] reject geometry end=({pend.x:.3f},{pend.y:.3f},{rad2deg(pend.yaw):.1f}) mans={mans} seq={seq_str(seq)}")
                continue

            pre_entries.append((s, pend, path, mans, seq))
            if debug:
                print(f"[{i:03d}] KEEP score={s:.3f} end=({pend.x:.3f},{pend.y:.3f},{rad2deg(pend.yaw):.1f}) mans={mans} seq={seq_str(seq)}")

        pre_entries.sort(key=lambda t: t[0])

        if debug:
            print("Accepted top pre-entry candidates:")
            for k, item in enumerate(pre_entries[:8], start=1):
                s, pend, _, mans, seq = item
                print(f" rank{k}: score={s:.3f} end=({pend.x:.3f},{pend.y:.3f},{rad2deg(pend.yaw):.1f}) mans={mans} seq={seq_str(seq)}")

        if not pre_entries:
            soft_candidates.sort(key=lambda t: t[0])
            pre_entries = soft_candidates[:6]

        if not pre_entries:
            m = middle_slot_metrics(self.env, start, self.spec)
            return PlannerResult(False, "no pre-entry candidate", "both_sides_v11(best_failed)", 0, [start], start, m)

        best_result: Optional[PlannerResult] = None
        best_score = float("inf")

        # deep-entry from top candidates
        for rank, (pscore, pre_pose, pre_path, mans0, pre_seq) in enumerate(pre_entries[:8], start=1):
            if debug:
                print(f"== Reverse-only deep entry from pre-entry rank {rank}: pose=({pre_pose.x:.3f},{pre_pose.y:.3f},{rad2deg(pre_pose.yaw):.1f}) ==")

            deep_lib = self.deep_entry_library(pre_pose)

            for j, dseq in enumerate(deep_lib, start=1):
                ok, dpath, pend = simulate_sequence(pre_pose, dseq, self.spec, self.env)
                if not ok:
                    continue

                full_path = pre_path + dpath[1:]
                strict, practical, m = self.validate_final(pend)
                score = self.deep_entry_score(pend)
                mans = mans0 + maneuver_count(dseq)

                if debug and j <= 420:
                    inside = m["inside_slot"]
                    print(f"deep[{j:03d}] KEEP score={score:.3f} end=({pend.x:.3f},{pend.y:.3f},{rad2deg(pend.yaw):.1f}) inside={inside} seq={seq_str(dseq)}")

                res = PlannerResult(
                    success=(strict or practical),
                    reason=("strict_success" if strict else "practical_success" if practical else "best_failed"),
                    planner=f"both_sides_v11({seq_str(pre_seq)} -> {seq_str(dseq)})",
                    maneuvers=mans,
                    path=full_path,
                    final_pose=pend,
                    metrics=m
                )

                if strict:
                    return res

                if score < best_score:
                    best_score = score
                    best_result = res

        if best_result is None:
            m = middle_slot_metrics(self.env, start, self.spec)
            return PlannerResult(False, "deep-entry failed", "both_sides_v11(best_failed)", 0, [start], start, m)

        return best_result


# ============================================================
# Test scene
# ============================================================

def make_both_sides_env() -> ParkingEnv:
    env = ParkingEnv()

    # slot depth = 1.39
    # parked fake cars on left and right slots
    # use simple slot-filling rectangles
    #
    # left slot x in [-1.14,-0.38], right slot x in [0.38,1.14]
    # parked cars approx width 0.67, length 1.30, centered inside those slots
    #
    # keep 45 mm front/rear nominal margin relative to slot depth:
    # y in [0.045, 1.345]
    env.left_obstacle = RectObstacle(
        x_min=-1.095,
        x_max=-0.425,
        y_min=0.045,
        y_max=1.345
    )
    env.right_obstacle = RectObstacle(
        x_min=0.425,
        x_max=1.095,
        y_min=0.045,
        y_max=1.345
    )
    return env


def print_result(name: str, res: PlannerResult):
    strict, practical, m = (
        (res.reason == "strict_success"),
        (res.reason == "strict_success" or res.reason == "practical_success"),
        res.metrics
    )

    print("SUCCESS:", res.success)
    print("REASON:", res.reason)
    print("PLANNER:", res.planner)
    print("MANEUVERS:", res.maneuvers)
    print("PATH POINTS:", len(res.path))
    print(f"FINAL POSE: x={res.final_pose.x:.4f}, y={res.final_pose.y:.4f}, yaw={rad2deg(res.final_pose.yaw):.3f} deg")
    print(f"  left_clear: {m['left_clear']:.4f}")
    print(f"  right_clear: {m['right_clear']:.4f}")
    print(f"  front_clear: {m['front_clear']:.4f}")
    print(f"  rear_clear: {m['rear_clear']:.4f}")
    print(f"  yaw_err_deg: {m['yaw_err_deg']:.4f}")
    print("STRICT_SUCCESS:", strict)
    print("PRACTICAL_SUCCESS:", practical)
    print("-" * 60)


# ============================================================
# Main
# ============================================================

def main():
    print("Standalone BOTH_SIDES planner v11")
    print("side-biased angled pre-entry + reverse-only deep-entry")
    print("Test starts: 3")
    print()

    env = make_both_sides_env()
    spec = VehicleSpec()
    planner = BothSidesPlannerV11(env, spec)

    starts = [
        Pose(0.000, 2.170, deg2rad(180.0)),
        Pose(-0.350, 2.090, deg2rad(177.0)),
        Pose(+0.350, 2.090, deg2rad(177.0)),
    ]

    strict_pass = 0
    practical_pass = 0

    for i, st in enumerate(starts, start=1):
        print("#" * 70)
        print(f"START {i:03d}: x={st.x:.3f}, y={st.y:.3f}, yaw={rad2deg(st.yaw):.3f} deg")
        res = planner.plan(st, debug=True)
        print_result(f"start{i}", res)

        if res.reason == "strict_success":
            strict_pass += 1
            practical_pass += 1
        elif res.reason == "practical_success":
            practical_pass += 1

    print(f"both_sides strict: {strict_pass}/3")
    print(f"both_sides practical: {practical_pass}/3")


if __name__ == "__main__":
    main()