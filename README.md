<!-- BlackRoad SEO Enhanced -->

# ulackroad physics engine

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad-Interactive](https://img.shields.io/badge/Org-BlackRoad-Interactive-2979ff?style=for-the-badge)](https://github.com/BlackRoad-Interactive)

**ulackroad physics engine** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

### BlackRoad Ecosystem
| Org | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | AI/ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh networking |

**Website**: [blackroad.io](https://blackroad.io) | **Chat**: [chat.blackroad.io](https://chat.blackroad.io) | **Search**: [search.blackroad.io](https://search.blackroad.io)

---


> 2D physics engine with rigid body dynamics and collision detection

Part of the [BlackRoad OS](https://blackroad.io) ecosystem — [BlackRoad-Interactive](https://github.com/BlackRoad-Interactive)

---

# blackroad-physics-engine

> 2D rigid body physics engine with AABB collision, restitution, friction, and projectile simulation.

Part of **BlackRoad-Interactive** — production game and graphics infrastructure.

## Features

| Module | Contents |
|--------|----------|
| `Vector2D` | +/-/*/dot/cross/magnitude/normalize/reflect/rotate/lerp |
| `AABB` | intersects/overlap/contains_point/expanded/moved |
| `RigidBody` | pos/vel/mass/restitution/friction/integrate/impulse |
| `PhysicsWorld` | add_body/step/check_collisions/resolve_collision/bounds |
| Trajectories | `projectile_path`, `range_equation`, `time_of_flight`, `max_height` |

## Quick Start

```python
from src.physics import PhysicsWorld, RigidBody, Vector2D, projectile_path

world = PhysicsWorld(gravity=9.81, bounds=(0, 0, 20, 20))
ball = world.add_body(RigidBody(
    pos=Vector2D(2, 2), vel=Vector2D(5, -3),
    mass=1.0, restitution=0.8
))

for _ in range(120):
    world.step(1/60)

print(f"Ball position: {ball.pos}")
print(f"Total KE: {world.total_kinetic_energy():.2f} J")

# Projectile
path = projectile_path(v0=20, angle_deg=45, steps=100)
```

## Run Demo

```bash
python src/physics.py
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## CI

GitHub Actions · Python 3.11 + 3.12 · pytest + flake8 + coverage
