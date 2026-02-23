"""Tests for BlackRoad Physics Engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import pytest
from physics import (
    Vector2D, AABB, RigidBody, Manifold, PhysicsWorld,
    projectile_path, range_equation, time_of_flight, max_height,
    visualise_trajectory, GRAVITY, EPSILON,
)


# ── Vector2D tests ────────────────────────────────────────────────────────────

class TestVector2D:
    def test_add(self):
        assert Vector2D(1, 2) + Vector2D(3, 4) == Vector2D(4, 6)

    def test_sub(self):
        assert Vector2D(5, 3) - Vector2D(2, 1) == Vector2D(3, 2)

    def test_mul_scalar(self):
        assert Vector2D(2, 3) * 2 == Vector2D(4, 6)

    def test_rmul(self):
        assert 3 * Vector2D(1, 2) == Vector2D(3, 6)

    def test_div(self):
        assert Vector2D(4, 6) / 2 == Vector2D(2, 3)

    def test_div_zero(self):
        with pytest.raises(ZeroDivisionError):
            Vector2D(1, 0) / 0

    def test_neg(self):
        assert -Vector2D(1, -2) == Vector2D(-1, 2)

    def test_dot(self):
        assert math.isclose(Vector2D(1, 0).dot(Vector2D(0, 1)), 0.0)
        assert math.isclose(Vector2D(1, 0).dot(Vector2D(1, 0)), 1.0)

    def test_cross(self):
        c = Vector2D(1, 0).cross(Vector2D(0, 1))
        assert math.isclose(c, 1.0)

    def test_magnitude(self):
        assert math.isclose(Vector2D(3, 4).magnitude, 5.0)

    def test_magnitude_sq(self):
        assert math.isclose(Vector2D(3, 4).magnitude_sq, 25.0)

    def test_normalize(self):
        n = Vector2D(3, 4).normalize()
        assert math.isclose(n.magnitude, 1.0)

    def test_normalize_zero(self):
        n = Vector2D(0, 0).normalize()
        assert n == Vector2D(0, 0)

    def test_reflect(self):
        v = Vector2D(1, -1)
        n = Vector2D(0, 1)
        r = v.reflect(n)
        assert math.isclose(r.x, 1.0) and math.isclose(r.y, 1.0)

    def test_rotate_90(self):
        v = Vector2D(1, 0).rotate(math.pi / 2)
        assert math.isclose(v.x, 0.0, abs_tol=1e-9)
        assert math.isclose(v.y, 1.0)

    def test_lerp(self):
        a, b = Vector2D(0, 0), Vector2D(10, 10)
        mid = a.lerp(b, 0.5)
        assert mid == Vector2D(5, 5)

    def test_perpendicular(self):
        p = Vector2D(1, 0).perpendicular()
        assert math.isclose(p.x, 0.0, abs_tol=1e-9) and math.isclose(p.y, 1.0)

    def test_equality(self):
        assert Vector2D(1.0, 2.0) == Vector2D(1.0, 2.0)
        assert Vector2D(1, 2) != Vector2D(1, 3)

    def test_repr(self):
        assert "Vector2D" in repr(Vector2D(1, 2))


# ── AABB tests ────────────────────────────────────────────────────────────────

class TestAABB:
    def test_intersects(self):
        a = AABB(Vector2D(0, 0), Vector2D(2, 2))
        b = AABB(Vector2D(1, 1), Vector2D(3, 3))
        assert a.intersects(b)

    def test_no_intersects(self):
        a = AABB(Vector2D(0, 0), Vector2D(1, 1))
        b = AABB(Vector2D(5, 5), Vector2D(6, 6))
        assert not a.intersects(b)

    def test_overlap_returns_vector(self):
        a = AABB(Vector2D(0, 0), Vector2D(2, 2))
        b = AABB(Vector2D(1, 0), Vector2D(3, 2))
        ov = a.overlap(b)
        assert ov is not None
        assert ov.magnitude > 0

    def test_overlap_none_when_apart(self):
        a = AABB(Vector2D(0, 0), Vector2D(1, 1))
        b = AABB(Vector2D(5, 5), Vector2D(6, 6))
        assert a.overlap(b) is None

    def test_center(self):
        a = AABB(Vector2D(0, 0), Vector2D(4, 4))
        assert a.center == Vector2D(2, 2)

    def test_from_center(self):
        a = AABB.from_center(Vector2D(5, 5), 1.0, 1.0)
        assert a.min == Vector2D(4, 4)
        assert a.max == Vector2D(6, 6)

    def test_contains_point(self):
        a = AABB(Vector2D(0, 0), Vector2D(4, 4))
        assert a.contains_point(Vector2D(2, 2))
        assert not a.contains_point(Vector2D(5, 5))

    def test_area(self):
        a = AABB(Vector2D(0, 0), Vector2D(3, 4))
        assert math.isclose(a.area, 12.0)

    def test_expanded(self):
        a = AABB(Vector2D(1, 1), Vector2D(3, 3))
        exp = a.expanded(1.0)
        assert exp.min == Vector2D(0, 0)
        assert exp.max == Vector2D(4, 4)

    def test_moved(self):
        a = AABB(Vector2D(0, 0), Vector2D(2, 2))
        m = a.moved(Vector2D(5, 5))
        assert m.min == Vector2D(5, 5)


# ── RigidBody tests ───────────────────────────────────────────────────────────

class TestRigidBody:
    def test_invalid_mass(self):
        with pytest.raises(ValueError):
            RigidBody(mass=-1)

    def test_static_inv_mass(self):
        b = RigidBody(is_static=True)
        assert b.inv_mass == 0.0

    def test_dynamic_inv_mass(self):
        b = RigidBody(mass=2.0)
        assert math.isclose(b.inv_mass, 0.5)

    def test_integrate_adds_gravity(self):
        b = RigidBody(pos=Vector2D(0, 0), vel=Vector2D(0, 0), mass=1.0, friction=0)
        b.integrate(1.0, gravity=10.0)
        assert b.vel.y > 0  # y-down convention

    def test_static_body_does_not_move(self):
        b = RigidBody(pos=Vector2D(5, 5), is_static=True)
        b.integrate(1.0, gravity=10.0)
        assert b.pos == Vector2D(5, 5)

    def test_apply_impulse(self):
        b = RigidBody(vel=Vector2D(0, 0), mass=1.0)
        b.apply_impulse(Vector2D(5, 0))
        assert math.isclose(b.vel.x, 5.0)

    def test_kinetic_energy(self):
        b = RigidBody(vel=Vector2D(2, 0), mass=2.0)
        assert math.isclose(b.kinetic_energy(), 4.0)

    def test_momentum(self):
        b = RigidBody(vel=Vector2D(3, 4), mass=2.0)
        p = b.momentum()
        assert math.isclose(p.x, 6.0) and math.isclose(p.y, 8.0)

    def test_aabb(self):
        b = RigidBody(pos=Vector2D(5, 5), half_w=1.0, half_h=1.0)
        assert b.aabb.contains_point(Vector2D(5, 5))


# ── PhysicsWorld tests ────────────────────────────────────────────────────────

class TestPhysicsWorld:
    def test_add_and_remove_body(self):
        w = PhysicsWorld()
        b = RigidBody()
        w.add_body(b)
        assert len(w.bodies) == 1
        assert w.remove_body(b)
        assert len(w.bodies) == 0

    def test_step_moves_body(self):
        w = PhysicsWorld(gravity=0)
        b = w.add_body(RigidBody(vel=Vector2D(10, 0), mass=1.0, friction=0))
        w.step(1.0)
        assert b.pos.x > 0

    def test_collision_callback(self):
        w = PhysicsWorld(gravity=0)
        hits = []
        w.on_collision(lambda m: hits.append(m))
        a = w.add_body(RigidBody(pos=Vector2D(0, 0), half_w=1, half_h=1, mass=1.0))
        b = w.add_body(RigidBody(pos=Vector2D(0.5, 0), half_w=1, half_h=1, mass=1.0))
        w.step(0.016)
        assert len(hits) > 0

    def test_bounds_enforcement(self):
        w = PhysicsWorld(gravity=0, bounds=(0, 0, 5, 5))
        b = w.add_body(RigidBody(pos=Vector2D(4.9, 2.5), vel=Vector2D(10, 0), mass=1.0, friction=0))
        w.step(1.0)
        assert b.pos.x <= 5.0

    def test_total_kinetic_energy(self):
        w = PhysicsWorld(gravity=0)
        w.add_body(RigidBody(vel=Vector2D(2, 0), mass=2.0, friction=0))
        assert w.total_kinetic_energy() > 0


# ── Projectile / trajectory tests ─────────────────────────────────────────────

class TestProjectile:
    def test_projectile_path_returns_points(self):
        path = projectile_path(20, 45, steps=50)
        assert len(path) > 0
        assert all(len(p) == 2 for p in path)

    def test_range_equation(self):
        r = range_equation(20, 45)
        assert r > 0

    def test_range_max_at_45(self):
        r30 = range_equation(20, 30)
        r45 = range_equation(20, 45)
        r60 = range_equation(20, 60)
        assert r45 > r30 and r45 > r60

    def test_time_of_flight_positive(self):
        t = time_of_flight(20, 45)
        assert t > 0

    def test_max_height_positive(self):
        h = max_height(20, 45)
        assert h > 0

    def test_visualise_trajectory(self):
        path = projectile_path(15, 45, steps=30)
        vis = visualise_trajectory(path)
        assert "+" in vis and "|" in vis

    def test_zero_angle(self):
        path = projectile_path(10, 0, steps=20)
        assert len(path) > 0
