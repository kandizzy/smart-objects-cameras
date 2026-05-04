#!/usr/bin/env python3
"""
Quick HTTP smoke test for the classroom API + orchestrator.

Start the API first in another terminal:
    CLASSROOM_API_KEY=testkey python classroom_api.py

Then run this:
    python smoke_test_api.py

Exits 0 if every check passes, 1 otherwise. Skips SSE (needs async);
everything else is plain requests.
"""

import os
import sys
import time

try:
    import requests
except ImportError:
    sys.stderr.write("requests not installed — pip install requests\n")
    sys.exit(2)


API_URL = os.getenv("CLASSROOM_API_URL", "http://localhost:8766")
API_KEY = os.getenv("CLASSROOM_API_KEY", "testkey")
AUTH = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# ── Output helpers ───────────────────────────────────────────────────────────

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

_passed = 0
_failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  {GREEN}✓{RESET} {label}" + (f" {DIM}({detail}){RESET}" if detail else ""))
    else:
        _failed += 1
        print(f"  {RED}✗{RESET} {label}" + (f" {RED}— {detail}{RESET}" if detail else ""))


def section(title: str):
    print(f"\n{BOLD}{title}{RESET}")


# ── Helpers ──────────────────────────────────────────────────────────────────


def get(path: str, **kwargs):
    return requests.get(f"{API_URL}{path}", timeout=5, **kwargs)


def post(path: str, payload: dict, headers: dict = None):
    return requests.post(
        f"{API_URL}{path}",
        json=payload,
        headers=headers or AUTH,
        timeout=5,
    )


def set_phase(phase: str) -> dict:
    r = post("/phase", {"phase": phase})
    r.raise_for_status()
    return r.json()


def push_state(**fields) -> dict:
    payload = {"camera_id": "smoke-test-camera", **fields}
    r = post("/push/state", payload)
    r.raise_for_status()
    return r.json()


# ── Tests ────────────────────────────────────────────────────────────────────


def test_health():
    section("1. Health / connectivity")
    try:
        r = get("/health")
    except requests.exceptions.ConnectionError:
        check(
            "API reachable at " + API_URL,
            False,
            "is `python classroom_api.py` running?",
        )
        return False

    check("GET /health returns 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json() if r.ok else {}
    check("health payload has 'status' field", data.get("status") == "ok")
    return r.ok


def test_phase_roundtrip():
    section("2. Phase state roundtrip")

    r = get("/phase")
    check("GET /phase returns 200", r.status_code == 200)
    data = r.json()
    check("phase payload has phase + started_at + duration_sec",
          {"phase", "started_at", "duration_sec"}.issubset(data.keys()))

    # Set to arrival so we have a known baseline
    result = set_phase("arrival")
    check("POST /phase arrival succeeds", result.get("phase") == "arrival")

    # Readback
    readback = get("/phase").json()
    check("GET /phase reflects new phase", readback["phase"] == "arrival",
          f"got {readback['phase']}")


def test_arrival_routing():
    section("3. ARRIVAL phase routing (person_change → broadcast)")

    # Reset the camera state so detect_changes has something to diff against.
    # First push establishes baseline.
    push_state(person_detected=False, person_count=0)

    # Now push a change — person_count transition should emit person_change.
    result = push_state(person_detected=True, person_count=2)
    routing = result.get("routing", {})
    check("POST /push/state returns routing counts",
          "broadcast" in routing and "ambient" in routing and "directed" in routing)
    check(
        "person_change during ARRIVAL is broadcast (not ambient)",
        routing.get("broadcast", 0) >= 1 and routing.get("ambient", 0) == 0,
        f"routing={routing}",
    )


def test_lecture_routing():
    section("4. LECTURE phase routing")

    set_phase("lecture")

    # Fresh baseline so only intended diffs emit
    push_state(person_count=10, fatigue_detected=False, whiteboard_text_detected=False)

    # Person traffic during lecture → AMBIENT
    r = push_state(person_count=15)
    routing = r.get("routing", {})
    check(
        "person_change during LECTURE routes to AMBIENT",
        routing.get("ambient", 0) >= 1,
        f"routing={routing}",
    )

    # Fatigue during lecture → DIRECTED to prof-dm
    r = push_state(fatigue_detected=True)
    routing = r.get("routing", {})
    check(
        "fatigue_change during LECTURE routes to DIRECTED",
        routing.get("directed", 0) >= 1,
        f"routing={routing}",
    )

    # Whiteboard during lecture → BROADCAST
    r = push_state(whiteboard_text_detected=True, whiteboard_text=["smoke-test notes"])
    routing = r.get("routing", {})
    check(
        "whiteboard_change during LECTURE routes to BROADCAST",
        routing.get("broadcast", 0) >= 1,
        f"routing={routing}",
    )


def test_policy_endpoint():
    section("5. Policy endpoint")

    r = get("/phase/policy")
    check("GET /phase/policy returns 200", r.status_code == 200)
    data = r.json()
    check("policy has current_phase + policies", "current_phase" in data and "policies" in data)

    policies = data.get("policies", {})
    expected_phases = {"unknown", "arrival", "lecture", "activity", "conclude", "departure"}
    check(
        "all six phases present",
        expected_phases.issubset(policies.keys()),
        f"missing: {expected_phases - set(policies.keys())}",
    )

    # Lecture policy should direct fatigue to prof-dm
    lecture_rules = policies.get("lecture", [])
    fatigue_rule = next(
        (r for r in lecture_rules if "fatigue_change" in r.get("event_types", [])),
        None,
    )
    check(
        "LECTURE policy directs fatigue_change → prof-dm",
        fatigue_rule is not None
        and fatigue_rule.get("salience") == "directed"
        and "prof-dm" in fatigue_rule.get("targets", []),
        f"rule={fatigue_rule}",
    )


def test_error_handling():
    section("6. Error handling")

    # Bad phase name → 400
    r = post("/phase", {"phase": "not-a-real-phase"})
    check("invalid phase returns 400", r.status_code == 400,
          f"got {r.status_code}")

    # Bad API key → 401
    r = post(
        "/phase",
        {"phase": "lecture"},
        headers={"X-API-Key": "wrong-key", "Content-Type": "application/json"},
    )
    check("invalid API key returns 401", r.status_code == 401,
          f"got {r.status_code}")


def test_cleanup():
    section("7. Cleanup")
    result = set_phase("unknown")
    check("reset to UNKNOWN phase", result.get("phase") == "unknown")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    print(f"{BOLD}Classroom API smoke test{RESET}")
    print(f"{DIM}target: {API_URL}{RESET}")
    start = time.time()

    if not test_health():
        print(f"\n{RED}Aborted — API not reachable.{RESET}")
        return 1

    try:
        test_phase_roundtrip()
        test_arrival_routing()
        test_lecture_routing()
        test_policy_endpoint()
        test_error_handling()
        test_cleanup()
    except requests.exceptions.HTTPError as e:
        print(f"\n{RED}HTTP error mid-test: {e}{RESET}")
        return 1
    except Exception as e:
        print(f"\n{RED}Unexpected error: {type(e).__name__}: {e}{RESET}")
        return 1

    elapsed = time.time() - start
    print()
    total = _passed + _failed
    if _failed == 0:
        print(f"{GREEN}{BOLD}{_passed}/{total} checks passed{RESET} {DIM}({elapsed:.2f}s){RESET}")
        return 0
    print(f"{RED}{BOLD}{_failed}/{total} checks failed{RESET} {DIM}({elapsed:.2f}s){RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
