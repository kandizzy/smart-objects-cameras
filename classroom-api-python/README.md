# Classroom API — Python orchestrator (reference implementation)

A small FastAPI server that implements the routing model the
[whitepaper](../docs/whitepaper/multi-agent-classroom.md) describes.
Written to be the simplest runnable thing that demonstrates the
architectural claims: a shared event bus, agent contracts, and
phase/context-aware routing of events into three categories —
**ambient**, **broadcast**, and **directed**.

This is the prototype the
[smart-classroom-story](https://github.com/kandizzy/smart-classroom-story)
companion site is written against.

A larger, more ambitious Node implementation lives in `../classroom-api/`.
The two systems are not in conflict — this one is the reference; that one
is the room runtime built on top of the same ideas, with different goals.

## What it does

- HTTP + Server-Sent Events bus on port `8766`
- Agent contracts in `classroom_api.py` — what each project consumes and emits
- `orchestrator.py` (~350 lines, pure Python, no FastAPI dependency) — the
  phase state machine and the routing decisions
- `mock_classroom_driver.py` — drives a full session through all phases so you
  can watch the categorisation work in real time
- `console.html` — instructor dashboard with phase controls, event stream,
  and project status
- Tests for the routing logic in `test_orchestrator.py` and `smoke_test_api.py`

## Run it

Install dependencies (one time):

```bash
python3 -m pip install -r requirements.txt
```

Start the server:

```bash
CLASSROOM_API_KEY=testkey python3 classroom_api.py
```

Open the dashboard at <http://localhost:8766/console>.

In a second terminal, run the mock driver to walk a full session:

```bash
CLASSROOM_API_URL=http://localhost:8766 \
CLASSROOM_API_KEY=testkey \
python3 mock_classroom_driver.py --speed 5
```

You'll see each event labeled with its routing category — for example,
`smart-stage:timer.offer -> gesture-timer routing={ambient: 0, broadcast: 0, directed: 1}`
is the directed-routing case the whitepaper points to as the orchestrator's
core job.

Run the routing tests directly:

```bash
python3 test_orchestrator.py
python3 smoke_test_api.py   # with the server running
```

## Relationship to other parts of the project

| Artifact | What it is |
|---|---|
| [`docs/whitepaper/multi-agent-classroom.md`](../docs/whitepaper/multi-agent-classroom.md) | The architectural argument |
| [`docs/whitepaper/diagrams/network-diagram.html`](../docs/whitepaper/diagrams/network-diagram.html) | The agent topology |
| `classroom-api-python/` (this folder) | Reference implementation of the routing model |
| `classroom-api/` | Node room runtime — timeline, project packets, dashboards |
| [`smart-classroom-story`](https://github.com/kandizzy/smart-classroom-story) | The narrative companion site |

## File map

- `classroom_api.py` — FastAPI server, agent contracts, SSE streams
- `orchestrator.py` — pure Python phase + routing logic
- `orchestrator_demo.py` — interactive walkthrough of the orchestrator
- `mock_classroom_driver.py` — full-session simulator
- `console.html` — instructor dashboard
- `student_template.py` / `student_template.html` — starting points for student projects
- `student_heartbeat.py` / `student_heartbeat.html` — minimal "I'm alive" check-in
- `test_orchestrator.py` — unit tests (no server needed)
- `smoke_test_api.py` — HTTP integration tests
- `supabase_schema.sql` — optional persistence layer
- `requirements.txt` — Python dependencies
