# Workstream: Orchestrator v1 — phase state + salience routing

**Status:** designed, not started
**Owner:** CK
**Depends on:** `tasks/classroom-api-detector-integration.md` (orchestrator routing rules are meaningless until fatigue/gaze/whiteboard events are on the bus)
**Related:** `docs/whitepaper/`, `journey_map.png`, `classroom-api/classroom_api.py`, `tui/smart_classroom_console.html`

---

## Context

The journey map's driving question is: *"When does the room know when to look at the room vs. output to an individual device?"* Today the classroom API answers every question the same way — every event fans out over SSE to every subscriber. That's a flat multi-agent system: everybody publishes, nobody coordinates, signal drowns in noise the moment fatigue + gaze + whiteboard detectors start pushing alongside person detection.

The missing piece is a **conductor**: a small module that holds session phase state (Arrival → Lecture → Activity → Conclude → Departure) and applies a per-phase routing policy to outgoing events. In multi-agent terms, this is the **supervisor agent** in a supervisor-specialist architecture. The cameras and student projects are specialists. The orchestrator is the one that routes.

Goal: ship a minimal orchestrator that (a) holds phase state, (b) classifies each outgoing event as ambient/broadcast/directed based on the current phase, and (c) exposes a tiny API for manual phase control. It's deliberately scoped small — no automatic phase detection, no config-polling directives, no per-subscriber preferences. Those are v2+.

---

## Design decisions

### 1. In-process module, not a new service

Create `classroom-api/orchestrator.py` as a module imported by `classroom_api.py`. One hook point in the `push_state` endpoint, between the event diffing and the SSE broadcast. **Not** a separate process, **not** a proxy in front of the API.

Rationale: a separate process would require an IPC mechanism, another port, another deploy target, and another failure mode. v1 needs to be legible and buildable in a day.

### 2. Three primitives, nothing more

Everything in v1 rests on three concepts:

- **Phase** — a named session state with a start time. Enum: `UNKNOWN`, `ARRIVAL`, `LECTURE`, `ACTIVITY`, `CONCLUDE`, `DEPARTURE`.
- **Salience** — how an event travels. Enum: `AMBIENT` (log only, no SSE), `BROADCAST` (SSE fan-out — current default behavior), `DIRECTED` (SSE to a named list of project IDs).
- **Rule** — a per-phase mapping from event types to salience.

The entire routing policy is declarative and fits on one screen. A student should be able to read `orchestrator.py` top-to-bottom and understand what the room does in each phase without tracing code.

### 3. Pass-through by default

When phase is `UNKNOWN` (the default) or the orchestrator is disabled, `route()` is a pure pass-through — every event becomes `BROADCAST`, which is today's behavior. This means:

- Shipping the orchestrator is a no-op until somebody calls `POST /phase`.
- Existing student projects and the Discord bot see zero behavior change until a phase is set.
- The orchestrator can be rolled out gradually and tested per-classroom.

### 4. Manual phase transitions only (v1)

Phase advances three ways in the design space:

1. **Manual** (v1) — Discord command `!phase lecture` + endpoint `POST /phase`. Fully deterministic.
2. **Auto-hinted** (v2) — use V-JEPA `predicted_class` as a *suggestion* ("looks like you're presenting — confirm Lecture?"). Human in the loop.
3. **Fully automatic** (later or never) — time + classification heuristics.

v1 does only #1. Rationale for the white paper: *"the conductor needs a baton; the baton being held by a human is a feature — the room responds to intent, not guesses."* That also happens to be the easiest thing to demo and debug.

### 5. One new event type

`phase_change` — emitted when `POST /phase` is called. **Always `BROADCAST`** regardless of policy (phase changes are always announced). This is the signal student projects subscribe to if they want to build phase-aware behavior — Seren's room dimming during Lecture, timer project resetting during Arrival, etc.

### 6. Persistence

Phase state persists to a new single-row-per-camera table `classroom_session` in Supabase so it survives API restarts. If Supabase is unavailable (local-only mode), fall back to a JSON file in `~/oak-projects/classroom_session.json`. Same fallback pattern the API already uses for other state.

---

## Data model (draft)

```python
# classroom-api/orchestrator.py
from enum import Enum
from dataclasses import dataclass, field

class Phase(str, Enum):
    UNKNOWN = "unknown"       # default, pass-through
    ARRIVAL = "arrival"
    LECTURE = "lecture"
    ACTIVITY = "activity"
    CONCLUDE = "conclude"
    DEPARTURE = "departure"

class Salience(str, Enum):
    AMBIENT = "ambient"       # written to log, NOT SSE-broadcast
    BROADCAST = "broadcast"   # SSE fan-out to all subscribers
    DIRECTED = "directed"     # SSE to specific project IDs only

@dataclass
class Rule:
    event_types: set[str]
    salience: Salience
    targets: list[str] = field(default_factory=list)  # project IDs for DIRECTED

POLICIES: dict[Phase, list[Rule]] = {
    Phase.ARRIVAL: [
        Rule({"person_change"}, Salience.BROADCAST),
        Rule({"fatigue_change"}, Salience.AMBIENT),    # nobody's tired yet
        Rule({"whiteboard_change"}, Salience.AMBIENT),
    ],
    Phase.LECTURE: [
        Rule({"person_change"}, Salience.AMBIENT),     # mid-lecture traffic = noise
        Rule({"fatigue_change"}, Salience.DIRECTED, targets=["prof-dm"]),
        Rule({"whiteboard_change"}, Salience.BROADCAST),
        Rule({"probe_classification"}, Salience.BROADCAST),
    ],
    Phase.ACTIVITY: [
        Rule({"person_change"}, Salience.BROADCAST),   # movement matters
        Rule({"fatigue_change"}, Salience.BROADCAST),
        Rule({"whiteboard_change"}, Salience.AMBIENT),
    ],
    Phase.CONCLUDE: [
        Rule({"whiteboard_change"}, Salience.BROADCAST),  # wrap-up notes matter
        Rule({"person_change"}, Salience.AMBIENT),
    ],
    Phase.DEPARTURE: [
        Rule({"person_change"}, Salience.BROADCAST),
    ],
    Phase.UNKNOWN: [],  # empty -> pass-through, nothing filtered
}
```

### Core function

```python
def route(events: list[dict], phase: Phase) -> RoutedEvents:
    """Classify each event by salience based on current phase policy.
    Returns a RoutedEvents container with .ambient, .broadcast, .directed attrs.
    If phase is UNKNOWN or no matching rule, event defaults to BROADCAST
    (pass-through behavior).
    """
```

---

## Hook point in classroom_api.py

One addition inside the existing `push_state` endpoint (lines 458–529), after event diffing and before SSE broadcast:

```python
# existing: build new state, compute events
events = compute_events(old_state, new_state)

# existing: persist events to log
supabase.table("classroom_events").insert(events).execute()

# NEW: orchestrator routes events by salience
routed = orchestrator.route(events, phase=orchestrator.current_phase())

# MODIFIED: SSE broadcast now respects routing
for subscriber in sse_subscribers:
    subscriber.send(routed.for_subscriber(subscriber.id))
```

`routed.for_subscriber(id)` returns:
- All `BROADCAST` events
- All `DIRECTED` events where `id` is in the targets list
- **No** `AMBIENT` events (those only live in the log)

---

## New API surface

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/phase` | Returns `{phase, started_at, duration_sec}` |
| `POST` | `/phase` | Body: `{phase: "lecture"}`. Emits `phase_change` event, updates `classroom_session` table |
| `GET` | `/phase/policy` | Returns the active routing rules as JSON — for debugging, TUI display, and white paper diagrams |

All three require the existing `X-API-Key` header auth pattern.

---

## Files to modify

**New files:**
- `classroom-api/orchestrator.py` — module with `Phase`, `Salience`, `Rule`, `POLICIES`, `route()`, `current_phase()`, `set_phase()` (~200 lines)
- `classroom-api/migrations/002_add_classroom_session.sql` — creates `classroom_session` table

**Edit:**
- `classroom-api/classroom_api.py`
  - Import orchestrator module
  - Add `GET /phase`, `POST /phase`, `GET /phase/policy` endpoints (~50 lines)
  - Modify `push_state` SSE broadcast to use `routed.for_subscriber(id)` (one block change)
  - Add `phase_change` to event type constants
- `classroom-api/supabase_schema.sql` — add `classroom_session` table definition to match migration
- `discord_bot.py` — add `!phase <name>` command that calls `POST /phase` (~30 lines)
- `tui/smart_classroom_console.html` — replace the faked phase dropdown with a real fetch of `GET /phase` and a `POST /phase` on change

**Not touched:**
- Any detector script — they don't need to know about phases. They keep pushing state as before; the orchestrator sits downstream of them.
- Existing event types or the room_mode computation — orthogonal to phase.

---

## Critical file references

| Purpose | File | Lines |
|---|---|---|
| Push endpoint to hook into | `classroom-api/classroom_api.py` | 458–529 |
| SSE broadcast logic | `classroom-api/classroom_api.py` | ~512–515 (broadcast loop) |
| Event type constants | `classroom-api/classroom_api.py` | event-emission block inside push_state |
| Schema file to keep in sync | `classroom-api/supabase_schema.sql` | — |
| Discord bot command pattern to copy | `discord_bot.py` | existing `!classroom` / `!mode` (lines 531–635) |
| TUI phase strip (currently fake) | `tui/smart_classroom_console.html` | top status bar block |

---

## Verification

1. **Pass-through regression:** With phase set to `UNKNOWN` (default after install), run all detectors and confirm SSE subscribers receive the same events they did before the orchestrator existed. This proves the rollout is safe.

2. **Manual phase walkthrough:** From the Discord bot, run `!phase arrival` → `!phase lecture` → `!phase activity` → `!phase conclude` → `!phase departure`. At each step, `curl http://localhost:8766/phase` returns the current phase, and a `phase_change` event lands in `/events` and on the SSE stream.

3. **Salience routing smoke test:** Set phase to `LECTURE`. Trigger a `person_change` event by walking in front of a camera. Confirm:
   - The event appears in `classroom_events` (log is complete regardless of salience).
   - The event does **NOT** appear on the SSE stream for a test subscriber (because Lecture routes person_change → AMBIENT).
   - A concurrent `whiteboard_change` event **does** appear on the same SSE stream (BROADCAST).

4. **Directed routing:** Set phase to `LECTURE`. Trigger a `fatigue_change` event. Confirm it's delivered only to the `prof-dm` target (or whatever subscriber IDs are in the rule) and not to other student projects.

5. **Policy endpoint:** `curl http://localhost:8766/phase/policy` returns the routing rules as JSON. This doubles as the source for the white paper's "how events route by phase" diagram.

6. **TUI integration:** Open `tui/smart_classroom_console.html`, confirm the phase strip at the top reads from `GET /phase`, and clicking a phase button issues `POST /phase` and the event log reflects the change in real time.

7. **Persistence:** Set phase to `LECTURE`. Restart the API. Confirm `GET /phase` still returns `LECTURE` after restart (loaded from Supabase or the JSON fallback).

---

## Out of scope (deliberately)

- **Automatic phase detection.** V-JEPA suggestion mode is v2.
- **Config-polling directives** — the orchestrator does NOT write to `camera_config.json` in v1. Detectors don't change their internal thresholds based on phase yet. That's a v2 feature.
- **Per-subscriber salience preferences.** Every student project gets the same phase-based routing in v1. "Seren's room wants everything, EchoDesk wants only room-mode changes" is v2.
- **Historical session replay.** No "play back what Lecture looked like yesterday" in v1.
- **Phase-aware Discord webhook routing.** Room-level webhook vs. per-user DMs stays hardcoded per detector in v1.
- **Multi-room orchestration.** v1 assumes one classroom / one phase state. Multi-room is a schema change (phase-per-room or phase-per-camera-group).

All tracked as follow-ups, not dependencies.

---

## What this unlocks immediately

- **The white paper's killer diagram** — a simple table showing how `fatigue_change` is routed AMBIENT → DIRECTED → BROADCAST across phases. One-glance answer to the journey map's red note about "when the room looks at the room vs. individual device."
- **The TUI's phase strip stops being faked** — it reads from `GET /phase` and controls `POST /phase`. The showcase demo becomes real instead of scripted.
- **Student projects can subscribe to `phase_change`** and build phase-aware behavior. Week 2 / Week 3 assignments in the white paper get meatier because students can design for specific phases.
- **Discord bot gets `!phase lecture`** — a satisfying live-demo moment that proves the orchestrator exists.
- **The whitepaper gains a concrete "here's the missing piece we designed" section** — a specifiable, bounded, buildable artifact instead of a vague "smart layer."

---

## Rough effort

- `orchestrator.py` module with data model + `route()` + persistence: ~2 hours
- API endpoints + Pydantic models + schema migration: ~1.5 hours
- Discord bot `!phase` command: ~30 min
- TUI phase strip wiring: ~30 min
- End-to-end verification (manual walkthrough): ~1 hour

**Total: ~5–6 hours of focused work**, sequenced *after* the detector integration workstream. The orchestrator's routing rules only mean something once fatigue/gaze/whiteboard events exist to route.
