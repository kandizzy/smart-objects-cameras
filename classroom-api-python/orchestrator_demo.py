"""
Orchestrator demo — walk through a fake class session.

Run this anywhere (no camera, no Supabase, no FastAPI):

    python classroom-api/orchestrator_demo.py            # scripted walkthrough
    python classroom-api/orchestrator_demo.py --interactive   # manual phase control
    python classroom-api/orchestrator_demo.py --policy   # print policy table only

The scripted mode simulates a class session by advancing through all phases
and firing plausible dummy events at each step, showing how the orchestrator
classifies each event (ambient / broadcast / directed) under the current
phase policy.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import orchestrator as orch
from orchestrator import Phase, Salience


# ── ANSI colors (no extra dependency) ────────────────────────────────────────

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
GREY = "\033[90m"


def color_for_salience(salience: str) -> str:
    return {
        Salience.BROADCAST.value: GREEN,
        Salience.AMBIENT.value: GREY,
        Salience.DIRECTED.value: MAGENTA,
    }.get(salience, RESET)


def badge(salience: str) -> str:
    c = color_for_salience(salience)
    label = {
        Salience.BROADCAST.value: "BROADCAST",
        Salience.AMBIENT.value: " AMBIENT ",
        Salience.DIRECTED.value: "DIRECTED ",
    }.get(salience, salience.upper())
    return f"{c}[{label}]{RESET}"


# ── Dummy event generator ────────────────────────────────────────────────────

DUMMY_CAMERA = "orbit"


def dummy(event_type: str, **payload) -> dict:
    return {
        "camera_id": DUMMY_CAMERA,
        "event_type": event_type,
        "payload": payload,
        "at": time.time(),
    }


# Per-phase scripted event batches. Each batch represents what might plausibly
# happen in ~15 seconds of the session.
SCRIPTED_BATCHES: dict[Phase, list[list[dict]]] = {
    Phase.ARRIVAL: [
        [
            dummy("person_change", old_count=0, new_count=1),
            dummy("whiteboard_change", text_detected=False, text=[]),
        ],
        [
            dummy("person_change", old_count=1, new_count=4),
            dummy("probe_classification", old_class=None, new_class="idle", confidence=0.61),
        ],
    ],
    Phase.LECTURE: [
        [
            dummy("person_change", old_count=4, new_count=15),  # late arrivals
            dummy("probe_classification", old_class="idle", new_class="presentation", confidence=0.88),
            dummy("whiteboard_change", text_detected=True, text=["Machine Learning 101"]),
        ],
        [
            dummy("fatigue_change", fatigue_detected=True),  # one student zoning out
            dummy("whiteboard_change", text_detected=True, text=["gradient descent"]),
        ],
    ],
    Phase.ACTIVITY: [
        [
            dummy("person_change", old_count=15, new_count=12),  # group movement
            dummy("fatigue_change", fatigue_detected=False),      # they woke up
            dummy("whiteboard_change", text_detected=True, text=["Exercise 1:"]),
        ],
        [
            dummy("probe_classification", old_class="presentation", new_class="discussion", confidence=0.79),
            dummy("anomaly_change", old_level=None, new_level="mild", score=0.35),
        ],
    ],
    Phase.CONCLUDE: [
        [
            dummy("whiteboard_change", text_detected=True, text=["Reading: Chapter 3"]),
            dummy("person_change", old_count=12, new_count=10),
        ],
    ],
    Phase.DEPARTURE: [
        [
            dummy("person_change", old_count=10, new_count=2),
            dummy("person_change", old_count=2, new_count=0),
        ],
    ],
}


# ── Rendering helpers ────────────────────────────────────────────────────────


def render_event_line(event: dict, salience: str) -> str:
    etype = event["event_type"]
    payload = event.get("payload", {})
    # Compact payload preview
    if payload:
        items = []
        for k, v in payload.items():
            if isinstance(v, list):
                v = "[" + ", ".join(str(x) for x in v[:2]) + ("..." if len(v) > 2 else "") + "]"
            items.append(f"{k}={v}")
        preview = " ".join(items)
    else:
        preview = ""
    return f"  {badge(salience)}  {CYAN}{etype:<22}{RESET} {DIM}{preview}{RESET}"


def render_batch(batch: list[dict], phase: Phase) -> None:
    routed = orch.route(batch, phase)
    for event in batch:
        etype = event["event_type"]
        if etype in orch.ALWAYS_BROADCAST:
            print(render_event_line(event, Salience.BROADCAST.value))
            continue
        rule = orch._rule_for_event(etype, phase)
        if rule is None:
            salience = Salience.BROADCAST.value
        else:
            salience = rule.salience.value
        line = render_event_line(event, salience)
        if rule and rule.salience is Salience.DIRECTED:
            line += f"  {MAGENTA}→ {', '.join(rule.targets)}{RESET}"
        print(line)
    c = routed.counts()
    print(
        f"  {DIM}subtotal: {c['ambient']} ambient · "
        f"{c['broadcast']} broadcast · {c['directed']} directed{RESET}"
    )


def render_phase_header(phase: Phase) -> None:
    bar = "─" * 68
    print()
    print(f"{BOLD}{YELLOW}{bar}{RESET}")
    print(f"{BOLD}{YELLOW}  PHASE → {phase.value.upper()}{RESET}")
    print(f"{BOLD}{YELLOW}{bar}{RESET}")


def render_phase_change_announcement(event: dict) -> None:
    payload = event.get("payload", {})
    old, new = payload.get("old_phase"), payload.get("new_phase")
    line = render_event_line(event, Salience.BROADCAST.value)
    print(line + f"  {DIM}({old} → {new}){RESET}")


def print_policy_table() -> None:
    snap = orch.policy_snapshot()
    print(f"\n{BOLD}Routing policy by phase:{RESET}\n")
    for phase, rules in snap.items():
        print(f"  {BOLD}{YELLOW}{phase.upper()}{RESET}")
        if not rules:
            print(f"    {DIM}(pass-through — every event defaults to BROADCAST){RESET}")
            continue
        for r in rules:
            types = ", ".join(r["event_types"])
            sal = r["salience"]
            badge_str = badge(sal)
            targets = ""
            if r["targets"]:
                targets = f" {MAGENTA}→ {', '.join(r['targets'])}{RESET}"
            print(f"    {badge_str}  {CYAN}{types}{RESET}{targets}")
        print()


# ── Scripted walkthrough ─────────────────────────────────────────────────────


def run_scripted(delay: float = 1.0) -> None:
    """Walk through every phase with dummy events."""
    print(f"{BOLD}{CYAN}🎭 ORCHESTRATOR DEMO — fake class session{RESET}")
    print(f"{DIM}No camera. No Supabase. Just dummy events and routing decisions.{RESET}")

    phase_order = [
        Phase.ARRIVAL,
        Phase.LECTURE,
        Phase.ACTIVITY,
        Phase.CONCLUDE,
        Phase.DEPARTURE,
    ]

    totals = {"ambient": 0, "broadcast": 0, "directed": 0}

    for phase in phase_order:
        pc_event = orch.set_phase(phase, persist_to=_tmp_state_path())
        render_phase_header(phase)
        render_phase_change_announcement(pc_event)

        for batch in SCRIPTED_BATCHES.get(phase, []):
            time.sleep(delay)
            routed = orch.route(batch, phase)
            render_batch(batch, phase)
            for k in totals:
                totals[k] += routed.counts()[k]

    print()
    print(f"{BOLD}Summary:{RESET} "
          f"{GREEN}{totals['broadcast']} broadcast{RESET}, "
          f"{GREY}{totals['ambient']} ambient{RESET}, "
          f"{MAGENTA}{totals['directed']} directed{RESET}")
    print(f"{DIM}(ambient events are in the log; they don't hit SSE subscribers){RESET}\n")


# ── Interactive mode ─────────────────────────────────────────────────────────


HELP_TEXT = f"""
{BOLD}Commands:{RESET}
  phase <name>   set current phase ({', '.join(p.value for p in Phase if p != Phase.UNKNOWN)})
  fire <type>    emit a dummy event of this type (e.g. 'fire fatigue_change')
  batch          fire the scripted batch for the current phase
  status         show current phase and duration
  policy         print the full routing policy table
  help           show this message
  quit           exit
"""


FIRE_TEMPLATES = {
    "person_change": lambda: dummy("person_change", old_count=3, new_count=4),
    "fatigue_change": lambda: dummy("fatigue_change", fatigue_detected=True),
    "whiteboard_change": lambda: dummy("whiteboard_change", text_detected=True, text=["dummy note"]),
    "probe_classification": lambda: dummy("probe_classification", old_class="idle", new_class="discussion", confidence=0.72),
    "anomaly_change": lambda: dummy("anomaly_change", old_level=None, new_level="mild", score=0.33),
}


def run_interactive() -> None:
    print(f"{BOLD}{CYAN}🎭 ORCHESTRATOR — interactive mode{RESET}")
    print(f"{DIM}Type 'help' for commands, 'quit' to exit.{RESET}\n")

    while True:
        status = orch.phase_status()
        prompt = (
            f"{BOLD}{YELLOW}[{status['phase']}]{RESET} "
            f"{DIM}({status['duration_sec']:.0f}s){RESET} > "
        )
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"quit", "exit", "q"}:
            break

        if cmd in {"help", "?"}:
            print(HELP_TEXT)
            continue

        if cmd == "status":
            print(f"  phase: {status['phase']}")
            print(f"  started_at: {status['started_at']:.0f}")
            print(f"  duration_sec: {status['duration_sec']:.0f}")
            continue

        if cmd == "policy":
            print_policy_table()
            continue

        if cmd == "phase":
            try:
                new_phase = Phase(arg.lower())
            except ValueError:
                print(f"  {RED}unknown phase: {arg}{RESET}")
                continue
            event = orch.set_phase(new_phase, persist_to=_tmp_state_path())
            print(f"  {GREEN}→ transitioned to {new_phase.value}{RESET}")
            render_phase_change_announcement(event)
            continue

        if cmd == "fire":
            if arg not in FIRE_TEMPLATES:
                print(f"  {RED}unknown event type: {arg}{RESET}")
                print(f"  options: {', '.join(FIRE_TEMPLATES.keys())}")
                continue
            event = FIRE_TEMPLATES[arg]()
            render_batch([event], orch.current_phase())
            continue

        if cmd == "batch":
            phase = orch.current_phase()
            batches = SCRIPTED_BATCHES.get(phase, [])
            if not batches:
                print(f"  {DIM}no scripted batch for phase {phase.value}{RESET}")
                continue
            for batch in batches:
                render_batch(batch, phase)
            continue

        print(f"  {RED}unknown command: {cmd}{RESET} (type 'help')")


# ── Temp state file (don't pollute ~/oak-projects in demos) ──────────────────

def _tmp_state_path():
    import tempfile, os
    return os.path.join(tempfile.gettempdir(), "orchestrator_demo_session.json")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Classroom orchestrator demo.")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="interactive prompt instead of scripted walkthrough")
    parser.add_argument("--policy", "-p", action="store_true",
                        help="just print the routing policy table and exit")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="seconds between batches in scripted mode (default: 0.8)")
    args = parser.parse_args()

    if args.policy:
        print_policy_table()
        return 0

    if args.interactive:
        run_interactive()
        return 0

    run_scripted(delay=args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
