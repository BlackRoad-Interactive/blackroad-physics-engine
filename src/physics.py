"""
BlackRoad Physics Engine — 2D Rigid Body Dynamics
Production physics with AABB collision, restitution, friction, and projectile simulation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ─── Constants ───────────────────────────────────────────────────────────────
GRAVITY: float = 9.81          # m/s²  (downward, positive-y-is-down convention)
EPSILON: float = 1e-9
MAX_ITER: int = 10             # position-correction iterations


# ─── Vector2D ────────────────────────────────────────────────────────────────

@dataclass
class Vector2D:
    x: float = 0.0
    y: float = 0.0

    # arithmetic
    def __add__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vector2D":
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vector2D":
        if abs(scalar) < EPSILON:
            raise ZeroDivisionError("Cannot divide vector by zero")
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vector2D":
        return Vector2D(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector2D):
            return NotImplemented
        return abs(self.x - other.x) < EPSILON and abs(self.y - other.y) < EPSILON

    # products
    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vector2D") -> float:
        """Scalar z-component of the 3D cross product."""
        return self.x * other.y - self.y * other.x

    # magnitude / normalisation
    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def magnitude_sq(self) -> float:
        return self.x ** 2 + self.y ** 2

    def normalize(self) -> "Vector2D":
        m = self.magnitude
        if m < EPSILON:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / m, self.y / m)

    def reflect(self, normal: "Vector2D") -> "Vector2D":
        """Reflect this vector about a unit normal."""
        return self - normal * (2 * self.dot(normal))

    def rotate(self, angle_rad: float) -> "Vector2D":
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )

    def lerp(self, other: "Vector2D", t: float) -> "Vector2D":
        return self + (other - self) * t

    def perpendicular(self) -> "Vector2D":
        return Vector2D(-self.y, self.x)

    def __repr__(self) -> str:
        return f"Vector2D({self.x:.4f}, {self.y:.4f})"


# ─── AABB ────────────────────────────────────────────────────────────────────

@dataclass
class AABB:
    """Axis-Aligned Bounding Box defined by min and max corners."""
    min: Vector2D = field(default_factory=lambda: Vector2D(0, 0))
    max: Vector2D = field(default_factory=lambda: Vector2D(1, 1))

    @classmethod
    def from_center(cls, center: Vector2D, half_w: float, half_h: float) -> "AABB":
        return cls(
            Vector2D(center.x - half_w, center.y - half_h),
            Vector2D(center.x + half_w, center.y + half_h),
        )

    @property
    def center(self) -> Vector2D:
        return Vector2D((self.min.x + self.max.x) / 2, (self.min.y + self.max.y) / 2)

    @property
    def half_extents(self) -> Vector2D:
        return Vector2D((self.max.x - self.min.x) / 2, (self.max.y - self.min.y) / 2)

    @property
    def width(self) -> float:
        return self.max.x - self.min.x

    @property
    def height(self) -> float:
        return self.max.y - self.min.y

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersects(self, other: "AABB") -> bool:
        return (
            self.min.x < other.max.x
            and self.max.x > other.min.x
            and self.min.y < other.max.y
            and self.max.y > other.min.y
        )

    def overlap(self, other: "AABB") -> Optional[Vector2D]:
        """Return the minimum overlap vector, or None if not overlapping."""
        if not self.intersects(other):
            return None
        ox = min(self.max.x, other.max.x) - max(self.min.x, other.min.x)
        oy = min(self.max.y, other.max.y) - max(self.min.y, other.min.y)
        if ox < oy:
            sign = 1 if self.center.x < other.center.x else -1
            return Vector2D(ox * sign, 0)
        else:
            sign = 1 if self.center.y < other.center.y else -1
            return Vector2D(0, oy * sign)

    def contains_point(self, p: Vector2D) -> bool:
        return self.min.x <= p.x <= self.max.x and self.min.y <= p.y <= self.max.y

    def expanded(self, margin: float) -> "AABB":
        return AABB(
            Vector2D(self.min.x - margin, self.min.y - margin),
            Vector2D(self.max.x + margin, self.max.y + margin),
        )

    def moved(self, delta: Vector2D) -> "AABB":
        return AABB(self.min + delta, self.max + delta)

    def __repr__(self) -> str:
        return f"AABB(min={self.min}, max={self.max})"


# ─── RigidBody ───────────────────────────────────────────────────────────────

@dataclass
class RigidBody:
    """A point-mass rigid body with AABB bounds."""
    pos: Vector2D = field(default_factory=Vector2D)
    vel: Vector2D = field(default_factory=Vector2D)
    mass: float = 1.0
    restitution: float = 0.5   # bounciness coefficient [0, 1]
    friction: float = 0.1      # kinetic friction coefficient
    half_w: float = 0.5
    half_h: float = 0.5
    is_static: bool = False     # infinite-mass immovable object
    gravity_scale: float = 1.0  # multiply global gravity

    def __post_init__(self) -> None:
        if self.mass <= 0 and not self.is_static:
            raise ValueError("Mass must be positive for dynamic bodies")

    @property
    def inv_mass(self) -> float:
        return 0.0 if self.is_static else 1.0 / self.mass

    @property
    def aabb(self) -> AABB:
        return AABB.from_center(self.pos, self.half_w, self.half_h)

    def apply_force(self, force: Vector2D, dt: float) -> None:
        if not self.is_static:
            self.vel = self.vel + force * (dt / self.mass)

    def apply_impulse(self, impulse: Vector2D) -> None:
        if not self.is_static:
            self.vel = self.vel + impulse * self.inv_mass

    def integrate(self, dt: float, gravity: float = GRAVITY) -> None:
        if self.is_static:
            return
        # Semi-implicit Euler
        grav_vec = Vector2D(0, gravity * self.gravity_scale)
        self.vel = self.vel + grav_vec * dt
        self.pos = self.pos + self.vel * dt
        # Damping (air resistance approximation)
        self.vel = self.vel * (1.0 - self.friction * dt)

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * self.vel.magnitude_sq

    def momentum(self) -> Vector2D:
        return self.vel * self.mass

    def __repr__(self) -> str:
        return f"RigidBody(pos={self.pos}, vel={self.vel}, mass={self.mass:.2f})"


# ─── Collision Manifold ───────────────────────────────────────────────────────

@dataclass
class Manifold:
    a: RigidBody
    b: RigidBody
    normal: Vector2D
    penetration: float

    @property
    def restitution(self) -> float:
        return min(self.a.restitution, self.b.restitution)


# ─── PhysicsWorld ─────────────────────────────────────────────────────────────

class PhysicsWorld:
    """
    Manages a collection of rigid bodies, steps the simulation, and resolves
    AABB collisions with restitution and positional correction.
    """

    def __init__(
        self,
        gravity: float = GRAVITY,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self.gravity = gravity
        self.bounds = bounds  # (min_x, min_y, max_x, max_y)
        self._bodies: List[RigidBody] = []
        self._collision_callbacks: List[Callable[[Manifold], None]] = []

    # ── Body management ────────────────────────────────────────────────────
    def add_body(self, body: RigidBody) -> RigidBody:
        self._bodies.append(body)
        return body

    def remove_body(self, body: RigidBody) -> bool:
        try:
            self._bodies.remove(body)
            return True
        except ValueError:
            return False

    def on_collision(self, callback: Callable[[Manifold], None]) -> None:
        self._collision_callbacks.append(callback)

    @property
    def bodies(self) -> List[RigidBody]:
        return list(self._bodies)

    # ── Simulation step ────────────────────────────────────────────────────
    def step(self, dt: float, substeps: int = 1) -> List[Manifold]:
        sub_dt = dt / substeps
        manifolds: List[Manifold] = []
        for _ in range(substeps):
            for body in self._bodies:
                body.integrate(sub_dt, self.gravity)
            self._enforce_bounds()
            manifolds.extend(self.check_collisions())
        return manifolds

    def check_collisions(self) -> List[Manifold]:
        manifolds: List[Manifold] = []
        for i, a in enumerate(self._bodies):
            for b in self._bodies[i + 1:]:
                m = self._narrow_phase(a, b)
                if m is not None:
                    manifolds.append(m)
                    self.resolve_collision(m)
                    for cb in self._collision_callbacks:
                        cb(m)
        return manifolds

    @staticmethod
    def _narrow_phase(a: RigidBody, b: RigidBody) -> Optional[Manifold]:
        ov = a.aabb.overlap(b.aabb)
        if ov is None:
            return None
        pen = ov.magnitude
        normal = ov.normalize() if pen > EPSILON else Vector2D(1, 0)
        return Manifold(a, b, normal, pen)

    def resolve_collision(self, m: Manifold) -> None:
        a, b, normal = m.a, m.b, m.normal
        rel_vel = b.vel - a.vel
        vel_along_normal = rel_vel.dot(normal)

        # Don't resolve if bodies are separating
        if vel_along_normal > 0:
            return

        e = m.restitution
        j = -(1 + e) * vel_along_normal / (a.inv_mass + b.inv_mass + EPSILON)
        impulse = normal * j

        a.apply_impulse(-impulse)
        b.apply_impulse(impulse)

        # Positional correction (Baumgarte slop)
        percent, slop = 0.2, 0.01
        correction_mag = max(m.penetration - slop, 0) / (a.inv_mass + b.inv_mass + EPSILON) * percent
        correction = normal * correction_mag
        if not a.is_static:
            a.pos = a.pos - correction * a.inv_mass
        if not b.is_static:
            b.pos = b.pos + correction * b.inv_mass

    def _enforce_bounds(self) -> None:
        if self.bounds is None:
            return
        min_x, min_y, max_x, max_y = self.bounds
        for body in self._bodies:
            if body.is_static:
                continue
            if body.pos.x - body.half_w < min_x:
                body.pos.x = min_x + body.half_w
                body.vel.x = abs(body.vel.x) * body.restitution
            if body.pos.x + body.half_w > max_x:
                body.pos.x = max_x - body.half_w
                body.vel.x = -abs(body.vel.x) * body.restitution
            if body.pos.y - body.half_h < min_y:
                body.pos.y = min_y + body.half_h
                body.vel.y = abs(body.vel.y) * body.restitution
            if body.pos.y + body.half_h > max_y:
                body.pos.y = max_y - body.half_h
                body.vel.y = -abs(body.vel.y) * body.restitution

    # ── Utilities ─────────────────────────────────────────────────────────
    def total_kinetic_energy(self) -> float:
        return sum(b.kinetic_energy() for b in self._bodies)


# ─── Trajectory Simulation ───────────────────────────────────────────────────

def projectile_path(
    v0: float,
    angle_deg: float,
    steps: int = 100,
    dt: float = 0.05,
    gravity: float = GRAVITY,
    y0: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Compute the trajectory of a projectile.

    Args:
        v0:        Initial speed (m/s)
        angle_deg: Launch angle in degrees above horizontal
        steps:     Number of time steps to compute
        dt:        Time increment per step (s)
        gravity:   Gravitational acceleration (m/s²)
        y0:        Initial height

    Returns:
        List of (x, y) tuples for each time step until ground impact.
    """
    angle_rad = math.radians(angle_deg)
    vx = v0 * math.cos(angle_rad)
    vy = -v0 * math.sin(angle_rad)  # negative = upward in screen-y convention

    positions: List[Tuple[float, float]] = []
    x, y = 0.0, y0

    for _ in range(steps):
        positions.append((x, y))
        x += vx * dt
        vy += gravity * dt
        y += vy * dt
        if y > y0 and x > 0:   # hit ground
            break

    return positions


def range_equation(v0: float, angle_deg: float, gravity: float = GRAVITY) -> float:
    """Analytical horizontal range on flat ground."""
    a = math.radians(angle_deg)
    return (v0 ** 2 * math.sin(2 * a)) / gravity


def time_of_flight(v0: float, angle_deg: float, gravity: float = GRAVITY) -> float:
    a = math.radians(angle_deg)
    return (2 * v0 * math.sin(a)) / gravity


def max_height(v0: float, angle_deg: float, gravity: float = GRAVITY) -> float:
    a = math.radians(angle_deg)
    return (v0 ** 2 * math.sin(a) ** 2) / (2 * gravity)


# ─── ASCII visualiser ────────────────────────────────────────────────────────

def visualise_trajectory(path: List[Tuple[float, float]], width: int = 60, height: int = 20) -> str:
    """Render a projectile path as an ASCII art string."""
    if not path:
        return ""
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1
    span_y = max_y - min_y or 1

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x, y in path:
        col = int((x - min_x) / span_x * (width - 1))
        row = int((y - min_y) / span_y * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = "·"
    # mark start and end
    grid[0][0] = "O"
    last_x, last_y = path[-1]
    last_col = int((last_x - min_x) / span_x * (width - 1))
    last_row = int((last_y - min_y) / span_y * (height - 1))
    grid[max(0, min(height - 1, last_row))][max(0, min(width - 1, last_col))] = "X"

    lines = ["+{:-<{w}}+".format("", w=width)]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+{:-<{w}}+".format("", w=width))
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== BlackRoad Physics Engine Demo ===\n")
    world = PhysicsWorld(gravity=GRAVITY, bounds=(0, 0, 20, 20))
    b1 = world.add_body(RigidBody(Vector2D(5, 2), Vector2D(3, -1), mass=2.0, restitution=0.7))
    b2 = world.add_body(RigidBody(Vector2D(10, 2), Vector2D(-2, 0.5), mass=1.0, restitution=0.8))
    static = world.add_body(RigidBody(Vector2D(10, 19), Vector2D(0, 0), mass=1.0, is_static=True, half_w=10, half_h=0.5))

    collisions = 0
    def count(m: Manifold) -> None:
        global collisions
        collisions += 1
    world.on_collision(count)

    for _ in range(120):
        world.step(1 / 60)

    print(f"Bodies: {len(world.bodies)}")
    print(f"Collisions resolved: {collisions}")
    print(f"Final KE: {world.total_kinetic_energy():.4f} J\n")

    path = projectile_path(20, 45)
    print(f"Projectile path points: {len(path)}")
    print(f"Range: {range_equation(20, 45):.2f} m")
    print(f"Max height: {max_height(20, 45):.2f} m\n")
    print(visualise_trajectory(path))
