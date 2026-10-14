# Ink Arena rules v1.1

These are the frozen v1.1 rules (v1 plus seeded randomness, see below). Every number below is a field of `inkarena.sim.Rules`
(`inkarena/sim/config.py`), and `DEFAULT_RULES` holds exactly these values. Distances are
in cells and times in ticks; there are 10 ticks per second.

## Match

| | |
|---|---|
| Format | 2v2 (`team_size = 2`) |
| Length | 600 ticks, 60 s (`ticks`) |
| Maps | 32x20 grids with walls, 180° rotationally symmetric, one base per team (`duel`, `zigzag`) |
| Win | More painted cells at the end. Equal counts are a draw. |

Team A's base is at the bottom of the map. Each team's bots see the map rotated so that
their own base is at the bottom (see [api.md](api.md)).

## Agents

| | |
|---|---|
| Body | Circle of radius 0.4 (`radius`), continuous position |
| HP | 3 (`max_hp`) |
| Ink tank | 100 (`max_ink`) |
| Turning | Facing turns toward `aim` by at most 30° per tick (`turn_rate`) |

## Movement

- Walking moves 0.25 cells per tick (`walk_speed`).
- Swimming means being submerged in your own ink. It works standing still or moving, and moving while submerged goes 0.5 cells per tick (`swim_speed`). `swim` is ignored off your own ink, and a submerged agent can't move off its own ink.
- On enemy ink, speed is halved (`enemy_ink_factor = 0.5`).
- Movement direction is independent of facing. Walls block movement.

## Ink and shots

- A shot costs 4 ink (`shot_cost`) and needs at least that much in the tank. After firing, an agent must wait 3 ticks before firing again (`fire_cooldown`).
- Submerged agents can't fire. If an action sets both `fire` and `swim`, fire wins and the agent surfaces.
- A submerged agent refills 5 ink per tick (`swim_refill`). An agent that isn't submerged refills 0.5 ink per tick (`idle_refill`) on any ground, starting 5 ticks after its last shot (`idle_refill_delay`).
- A shot is a projectile that moves 1.5 cells per tick (`shot_speed`) for up to 3 cells (`shot_range`), along the shooter's facing give or take a random spread of up to 3° (`shot_spread`).
- It paints every cell whose center is within 1.0 of its path, a strip 3 cells wide (`strip_halfwidth`). Where it stops, it paints a splat of radius 1.5 (`splat_radius`).
- Walls stop shots. Splats and death bursts never reach a cell that a wall hides from their center.
- A shot that hits an enemy deals 1 damage (`shot_damage`) and stops there.

## Randomness

All randomness comes from one RNG seeded with the match seed, so a seed plus the actions
still reproduces a match exactly. Added in v1.1 so that two bots that ignore their seeds
don't replay the same game in every match.

- **Spawn jitter.** At the start of each match every spawn point moves by a random offset of up to 1 cell on each axis (`spawn_jitter`). Team B's offset is the mirror image of team A's, so the map stays fair. An offset that would put an agent in a wall is redrawn. Agents respawn at the same jittered point for the whole match.
- **Shot spread.** Each shot leaves at the shooter's facing plus a random angle of up to 3° either way (`shot_spread`).

## Health, death and respawn

- After 30 ticks without taking damage (`regen_delay`), an agent regains 1 HP every 20 ticks (`regen_interval`).
- At 0 HP the agent is splatted. It bursts into the killer's color with radius 2 (`burst_radius`) and respawns at its base after 30 ticks (`respawn_delay`).
- For 20 ticks after respawning (`respawn_invisible`), an agent can't be seen by enemies. It can still be hit.

## Visibility

An enemy is visible to a team when at least one of its living agents has it:

- inside its view cone, 120° wide and centered on its facing (`view_half_angle = 60°`),
- within 6 cells (`view_range`),
- and in line of sight, with no wall cell between them.

On top of that, a submerged enemy must be within 2 cells (`submerged_view_range`), and an
enemy in respawn invisibility is never visible. Enemy projectiles follow the same rule.
Teammates and the whole ink map are always known.

## Tick order

Each tick runs these steps in order. Agents don't collide with each other, and hits and
paint resolve all at once, so the order of agents never matters.

1. Read actions.
2. Turn facing toward `aim`.
3. Move.
4. Fire.
5. Advance projectiles and resolve hits.
6. Apply damage, then deaths.
7. Paint, all at once.
8. Check whether each submerged agent is still on its own ink.
9. Refill ink and regenerate HP.
10. Respawn.
11. Update the score and visibility.

## Sandbox

- Each bot runs in its own subprocess, with no network and 1 GiB of memory.
- Building the `Policy` may use up to 1 s of CPU.
- Each tick allows 50 ms hard, and 5 ms on average over the match after a 0.5 s head start.
- Going over the hard limit, or raising, is a crash. It makes that tick a no-op for the team and restarts the bot. A bot over its average budget gets a no-op tick but no crash.
- After 3 crashes (`max_crashes`), the team forfeits the match.

## How these values were picked

The values were tuned with `tools/tune.py`. It plays baseline against baseline under
variants of the rules, 50 seeds on each map. With these values:

- 96% of the map gets painted.
- The final margin averages 15% of the painted cells.
- The lead changes hands about 8 times per match.
- Cells in the middle third flip between teams about 650 times per match.
- There are about 10 splats per match.
- Sides are balanced: 48 wins for side A, 51 for side B and 1 draw.

The other variants tried were cheaper or costlier shots, faster refill, smaller paint,
faster walking, faster respawn, and longer view or shot range. None of them changed these
numbers enough to justify moving off the starting values.
