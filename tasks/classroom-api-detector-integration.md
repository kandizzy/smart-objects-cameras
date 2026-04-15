# Workstream: Wire fatigue, gaze, and whiteboard detectors to the classroom API

**Status:** planned, not started
**Owner:** CK
**Related:** `docs/whitepaper/`, `tui/smart_classroom_console.html`, `classroom-api/`

---

## Context

The classroom API (`classroom-api/classroom_api.py`) is a FastAPI + Supabase service that already accepts detector state via `POST /push/state`, fans it out over SSE to student projects and the TUI, and computes a reactive `room_mode`. Only `person_detector.py` currently pushes to it. The other three detectors — `fatigue_detector.py`, `gaze_detector.py`, `whiteboard_reader.py` / `whiteboard_reader_full.py` — still write to local JSON files only.

This gap blocks three downstream efforts at once:

1. **The TUI console** (`tui/smart_classroom_console.html`) — needs live data from all cameras to stop running on mock data.
2. **The white paper** (`docs/whitepaper/`) — needs real classroom-session logs and screenshots showing cross-detector signals (fatigue + gaze + whiteboard) to tell an honest story.
3. **The orchestrator layer** — can't be built until every signal is on the bus. The journey map's "ambient vs. explicit" split is meaningless if half the ambient signals never reach the API.

Goal: get fatigue, gaze, and whiteboard data flowing through `/push/state` with the same reliability pattern as `person_detector.py`, plus a minimal schema migration to make gaze and richer fatigue data persistable.

---

## Design decisions

### 1. Shared push helper — new file

The current pattern in `person_detector.py` (lines 85–105) is ~20 lines of inline `requests.post(...)` with try/except and env var lookup. Duplicating this across four more detectors is a maintenance drag. Create `utils/classroom_api_client.py` with a single function:

```python
def push_state(camera_id: str, **fields) -> None:
    """Fire-and-forget POST to classroom API. Silently swallows errors.
    Reads CLASSROOM_API_URL and CLASSROOM_API_KEY from env.
    No-op if URL is unset (local-only mode)."""
```

- Sync, blocking, 2s timeout — matches `person_detector.py` behavior (line 102).
- Bare `except Exception: pass` — matches `person_detector.py` line 104. Never block the detection loop.
- No-op when `CLASSROOM_API_URL` is empty so students running detectors offline still work.
- **Do not refactor `person_detector.py` in this pass.** Keep blast radius small; adopt the helper for new detectors only. Person can migrate in a follow-up.

### 2. Schema migration — additive, nullable, one file

The current `classroom_state` table (in `classroom-api/supabase_schema.sql`) has `fatigue_detected` (bool) and `whiteboard_text*` fields, but **zero** gaze fields and **no** EAR/head pose detail. Integrating gaze without schema changes means pushing nothing useful — the detector's reason for existing is the gaze vector and head pose.

Add a migration file `classroom-api/migrations/001_add_fatigue_gaze_fields.sql`:

```sql
ALTER TABLE classroom_state
  ADD COLUMN IF NOT EXISTS fatigue_ear_left REAL,
  ADD COLUMN IF NOT EXISTS fatigue_ear_right REAL,
  ADD COLUMN IF NOT EXISTS fatigue_eyes_closed_pct REAL,
  ADD COLUMN IF NOT EXISTS fatigue_head_tilted_pct REAL,
  ADD COLUMN IF NOT EXISTS gaze_direction TEXT,
  ADD COLUMN IF NOT EXISTS gaze_x REAL,
  ADD COLUMN IF NOT EXISTS gaze_y REAL,
  ADD COLUMN IF NOT EXISTS gaze_z REAL,
  ADD COLUMN IF NOT EXISTS head_yaw REAL,
  ADD COLUMN IF NOT EXISTS head_pitch REAL,
  ADD COLUMN IF NOT EXISTS head_roll REAL;
```

All columns nullable, all additive — safe to run on a live Supabase with existing data. Apply by pasting into the Supabase SQL editor (or `psql`). Also update `supabase_schema.sql` itself to reflect the new columns so a fresh deployment matches.

### 3. Pydantic model — mirror the migration

Extend `PushStateRequest` in `classroom-api/classroom_api.py` (lines 110–124) with the same field names as the SQL migration, all `Optional` and defaulting to `None`. Keep existing fields untouched. The state merge logic (lines 470–477) already does dict merging, so new fields flow through automatically without endpoint logic changes.

**No new event types.** Fatigue, whiteboard, and room-mode events already exist. Gaze can piggyback on `anomaly_change` style later if needed — out of scope for this pass.

### 4. Fix gaze detector head pose extraction (known half-built code)

From research: `gaze_detector.py` lines 345–347 stub `head_yaw/pitch/roll` as `last_head_yaw/pitch/roll`, which are initialized to 0.0 and never updated. The head pose values exist inside the neural network output but are dropped before reaching the status dict. **This is in scope** because pushing zeros to the API would be worse than not pushing at all — it would poison the data.

Action: read the `utils/process_keypoints.py` helper (referenced in CLAUDE.md) and the head pose concatenation in `utils/host_concatenate_head_pose.py` to find where yaw/pitch/roll are computed, then plumb them through to the main loop's `last_head_*` variables so they actually update each frame.

Time-box this: if the plumbing turns out to be more than ~30 minutes of work, push `gaze_direction` only and open a separate issue for head pose. Don't let this block the other integrations.

### 5. Whiteboard — integrate `whiteboard_reader_full.py` only

`whiteboard_reader.py` has a placeholder `decode_text()` that returns `"[Text detected - see visualization]"` (lines 140–149). Pushing that string to the API would be noise. **Scope restriction:** only `whiteboard_reader_full.py` pushes `whiteboard_text` content. `whiteboard_reader.py` can still push `whiteboard_text_detected: true/false` (the detection-only flag) so the TUI knows something is there, but no text array.

### 6. Push cadence — match person_detector

Same pattern as `person_detector.py` (lines 283 and 305):
- Push immediately when state transitions (after debounce confirms the change).
- Push every 10 seconds as a heartbeat regardless of state, so the API knows the detector is running and `updated_at` stays fresh.

Use a module-level constant `STATUS_UPDATE_INTERVAL = 10.0` in each detector if not already present.

---

## Files to modify

**New files:**
- `utils/classroom_api_client.py` — shared push helper (~30 lines)
- `classroom-api/migrations/001_add_fatigue_gaze_fields.sql` — additive schema migration

**Edit:**
- `classroom-api/supabase_schema.sql` — add the 11 new columns to keep fresh deploys in sync
- `classroom-api/classroom_api.py` — extend `PushStateRequest` model (lines 110–124) with 11 optional fields
- `fatigue_detector.py` — import helper, call `push_state(...)` at the two points where status JSON is written (lines 313–318 and 356–365)
- `gaze_detector.py` — (a) fix head pose plumbing to surface yaw/pitch/roll from NN output, (b) import helper and call `push_state(...)` at line 370–378 write point
- `whiteboard_reader.py` — import helper and call `push_state(whiteboard_text_detected=...)` only (no text content) at lines 404–407 and 450–455
- `whiteboard_reader_full.py` — import helper and call `push_state(whiteboard_text=..., whiteboard_text_detected=True)` at the equivalent points

**Not touched (explicitly):**
- `person_detector.py` — keeps its inline push to minimize blast radius. Migration to shared helper is a follow-up.
- `discord_bot.py` — it reads `/state` and `/mode`; new fields flow through without bot changes.

---

## Critical file references

| Purpose | File | Lines |
|---|---|---|
| Push helper pattern to replicate | `person_detector.py` | 85–105 (function), 283 & 305 (call sites) |
| API endpoint to POST against | `classroom-api/classroom_api.py` | 458–529 |
| Pydantic model to extend | `classroom-api/classroom_api.py` | 110–124 |
| Schema to migrate | `classroom-api/supabase_schema.sql` | 12–43 |
| Fatigue status write point | `fatigue_detector.py` | 313–318, 356–365 |
| Gaze status write point | `gaze_detector.py` | 370–378 |
| Gaze head pose stub (bug to fix) | `gaze_detector.py` | 345–347 |
| Whiteboard (detection-only) write | `whiteboard_reader.py` | 404–407, 450–455 |
| Whiteboard full-OCR state variable | `whiteboard_reader_full.py` | 74 (`last_confirmed_text`) |

---

## Verification

End-to-end, without needing a live Supabase:

1. **Unit sanity (local-only mode):** Unset `CLASSROOM_API_URL`. Run each detector on a Pi. Confirm nothing breaks — `push_state()` should no-op silently. This proves the helper is safe for offline use.

2. **Local API roundtrip:** On your laptop or one Pi, run `python classroom-api/classroom_api.py` without Supabase creds (local-only mode, uses in-memory dict). Set `CLASSROOM_API_URL=http://localhost:8766` and `CLASSROOM_API_KEY=<whatever>` in the detector's shell. Run each detector, then `curl http://localhost:8766/state` and confirm the expected fields appear for each `camera_id`.

3. **Event emission:** Trigger a state change (close your eyes for fatigue; look left for gaze; write on a whiteboard). Confirm the API logs an event and `curl http://localhost:8766/events` returns the new event. This validates that state diffing works for the new fields.

4. **SSE stream:** Open `tui/smart_classroom_console.html` in a browser pointed at the local API and watch the event log populate in real time. This is both a smoke test for the backend AND your first integration point for the white paper.

5. **Supabase sanity:** Apply the migration on the live Supabase instance, run one detector against the real API URL, check that the new columns populate in the `classroom_state` row for that camera.

6. **Regression check on person_detector:** Run `person_detector.py` against the upgraded API and confirm its existing push still works unchanged. (The Pydantic model additions are all optional; this should be a no-op.)

---

## Out of scope (deliberately)

- Refactoring `person_detector.py` to use the shared helper.
- Adding gaze to the `room_mode` priority computation in `classroom_api.py`.
- New Discord bot commands exposing fatigue/gaze.
- Phase state machine / orchestrator layer.
- Fixing `whiteboard_reader.py`'s placeholder OCR (use `whiteboard_reader_full.py` instead).

These are tracked as follow-ups, not dependencies.

---

## Rough effort

- Shared helper + schema migration + Pydantic extension: ~1 hour
- Fatigue detector integration: ~30 min
- Whiteboard full integration: ~30 min
- Whiteboard detection-only integration: ~15 min
- Gaze detector integration including head pose plumbing: ~1–2 hours (the unknown)
- End-to-end verification: ~30 min

**Total: ~4–5 hours of focused work**, with the gaze head pose fix as the biggest risk. If head pose plumbing balloons, drop it to `gaze_direction` only and ship the rest.
