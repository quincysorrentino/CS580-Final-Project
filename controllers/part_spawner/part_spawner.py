"""Supervisor: spawn random parts at the start of the conveyor.

The three parts in the world (PART_BOTTLE, PART_CLOCK, PART_CAT) are used as
templates: on Play they are copied and removed. After that, a random part
(duplicates allowed) is spawned every few seconds at a random sideways
position, and parts are deleted once they near the end of the belt.
"""

import random

from controller import Supervisor

SPAWN_X = -1.8          # belt runs x = -2..2 and moves toward +X
END_X = 1.5             # delete parts here (bottle sticks out 0.3 m past its origin)
Y_RANGE = (-0.1, 0.1)   # random sideways offset on the 0.5 m wide belt
GAP_S = (3.0, 6.0)      # random time between spawns, seconds
CLEAR = 0.3             # wait if a part is still this close to the spawn point (m)

sup = Supervisor()
dt = int(sup.getBasicTimeStep())
children = sup.getRoot().getField("children")

# Copy each template part (as VRML text + its height), then remove it.
templates = []
for def_name in ("PART_BOTTLE", "PART_CLOCK", "PART_CAT"):
    node = sup.getFromDef(def_name)
    vrml = node.exportString().removeprefix(f"DEF {def_name} ")
    z = node.getField("translation").getSFVec3f()[2]
    templates.append((def_name, vrml, z))
    node.remove()

parts = []
next_spawn = 0.0

while sup.step(dt) != -1:
    t = sup.getTime()

    for p in parts[:]:
        if p.getPosition()[0] > END_X:
            p.remove()
            parts.remove(p)

    if t < next_spawn or any(abs(p.getPosition()[0] - SPAWN_X) < CLEAR for p in parts):
        continue

    def_name, vrml, z = random.choice(templates)
    y = random.uniform(*Y_RANGE)
    count = children.getCount()
    children.importMFNodeFromString(-1, vrml)
    if children.getCount() == count:          # import failed (error printed by Webots)
        next_spawn = t + 1.0
        continue
    node = children.getMFNode(-1)
    node.getField("translation").setSFVec3f([SPAWN_X, y, z])
    parts.append(node)
    print(f"[spawner] t={t:.2f} spawned {def_name} y={y:+.3f}", flush=True)

    next_spawn = t + random.uniform(*GAP_S)
