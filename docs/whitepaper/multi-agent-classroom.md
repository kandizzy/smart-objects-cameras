# The Classroom as a Multi-Agent System

This system was always designed around a single premise: cameras should be conversational agents, not passive sensors. You talk to them, they talk back, they talk to each other. The classroom isn't monitored — it's inhabited by software that participates. That's a multi-agent system, even if we never used the term.

What follows is a framework for making that architecture legible — naming the agents, mapping their coordination patterns, and scoping the piece we've been designing toward all semester: the orchestrator.

---

## The Agents Already Running

A multi-agent system, stripped to essentials, is: multiple independent actors with their own loops, their own state, and their own goals, coordinating through a shared bus. This classroom has at least six kinds:

| Agent | Loop | State | Goal | Channel |
|-------|------|-------|------|---------|
| **Camera-detectors** (person, fatigue, gaze, whiteboard) | Perception loop @ 5–30 FPS | Local status files + (soon) API state | "Report what I see" | POST /push/state |
| **Discord bot** | Command event loop | Reads API state | "Answer humans when asked" | Discord commands + webhook |
| **Student projects** (Tony, EchoDesk, timer, Seren's room, Gus, ...) | SSE subscription loop | Own project events | "React to room in my specific way" | GET /subscribe/state, POST /projects/:id/events |
| **V-JEPA probe** | Classification loop | Prediction history | "Label what's happening" | predicted_class field via push |
| **Humans in the room** (prof, students) | Biological | Memory, intent | Varies per person | Physical presence + Discord |
| **The orchestrator** (not yet built) | Context state machine | Shared room context | "Conduct the room" | Would own /mode as a writable endpoint |

The classroom API functions as a **message bus with a shared memory model**. SSE is pub/sub. The `classroom_events` table is the event log. The `student_projects` registry is the agent directory. These were the first things we built, because the system needed a way for independent actors to find each other and share state.

---

## What Makes This Novel

Most multi-agent work right now — AutoGen, CrewAI, LangGraph, MCP-based systems — assumes all agents are software talking to software. This classroom puts three fundamentally different kinds of agents on the same bus:

1. **Perception agents** — cameras that sense physical reality. No language, no reasoning, just continuous signal at frame rate.
2. **Reactive agents** — student projects that respond to room events. Some are LLM-backed, some aren't. A p5.js sketch that changes color when `room_mode` flips is still an agent by any meaningful definition.
3. **Human agents** — the students and professor physically in the room, whose goals the system exists to serve.

This mixing is the point. The design question was never "how do we get software agents to collaborate" — it was "how do we build a coordination layer where cameras, code, and people coexist as participants in the same space." That's a different and harder problem than what the current multi-agent frameworks are solving, and it's why the system looks the way it does.

---

## The Routing Problem at the Center

The journey map's central question — *"when does the room look at the room vs. output to individual devices?"* — is a multi-agent communication routing problem. Given N agents producing events and M agents consuming them, which events should be:

- **Ambient** — whispered on the bus for any subscriber to pick up
- **Explicit** — surfaced as a notification to a defined audience
- **Directed** — sent to one specific agent

In multi-agent literature this is called **addressing and salience routing**, and it's largely unsolved for mixed populations of human and software agents. The classroom is a live testbed for it.

---

## Context: The Thing Routing Depends On

Routing decisions only make sense if someone knows what the room is doing. The same fatigue reading means one thing during a forty-five-minute lecture and something completely different during a five-minute break. A whiteboard reading is content when the professor is teaching and garbage when they're erasing. A person-count spike matters during quiet study time and is just clutter during open discussion.

None of those signals changed. The **context** around them did.

Context is the situational frame that gives events their meaning. It has several dimensions:

- **Temporal** — what phase of the session we're in (lecture, break, discussion, work time)
- **Social** — who's in the room, who's speaking, who's leading, who's listening
- **Task** — what the room is collectively *doing* right now
- **Historical** — what just happened, and whether the current moment is a continuation or a shift

A flat multi-agent system has no place to hold context. Every detector publishes its own narrow state — "I see 12 people," "EAR dropped below threshold," "text region detected" — and every subscriber has to guess at the situational frame from scratch, using only what it can see. That guessing is the work the orchestrator should be doing once, for everyone.

This is what the journey map was really asking: *how does the room know what it is right now, and how does that knowing travel to the pieces that need it?*

---

## The Orchestrator: What Happens Along the Arrows

The orchestrator isn't a missing piece — it's the piece we've been building toward. The semester prioritized the agents and the bus first: get the detectors running, get the API stable, get student projects subscribed and reacting. You can't design coordination until you know what's being coordinated.

In recent whiteboard sessions, the design question kept surfacing: *what happens along the arrows between student projects and an overall intelligence?* Every time we drew a line from a detector to a project, or from a project to the room state, the same gap appeared — who decides what travels along that line, when, and at what volume? Naming context as the missing input makes the answer precise: **the orchestrator is the agent that holds context and uses it to route.**

Right now the system is flat — every agent publishes, every agent subscribes, and each one tries to reconstruct the room's context on its own. That works with two or three agents. With six kinds on the bus, it starts to strain, because every project is guessing at the same frame and arriving at different answers.

The orchestrator is the conductor agent: the one that knows *"the room is in a lecture context right now — presenter at the board, students seated and facing, no break for another ten minutes — so elevate gaze and whiteboard signals, suppress person-count chatter, route fatigue alerts as DMs to the prof only, and never interrupt the presenter."* Same role as a supervisor agent in CrewAI or a router node in LangGraph — just operating across physical sensors and human co-agents instead of LLM chains.

This also clarifies what the student projects are in architectural terms: **specialist agents in a supervisor-specialist pattern**, where the supervisor is the next thing to build. That's why every project currently subscribes to everything and filters locally — the routing layer they've been sketching on the whiteboard isn't running yet, so each project is inventing its own partial version of the room context.

### What the orchestrator actually needs

Framed as a supervisor agent, the scope becomes concrete:

- **(a)** A **context state** — what the room is doing right now, across temporal, social, task, and historical dimensions
- **(b)** A **context-aware routing policy** — which signals reach which subscribers, at what salience, given the current context
- **(c)** A way to **push context to specialist agents** so they don't have to guess (the config-polling mechanism already exists for this)

Three pieces, not a vague "smart layer."

---

## The Risk Worth Naming

The failure mode to watch for isn't too much data — it's **data without context**. When every detector publishes its reading on its own schedule, with no shared sense of what the room is doing, the stream becomes a series of numbers that are technically true but practically meaningless. A fatigue score. A person count. A whiteboard read. A gaze direction. Any one of them might matter right now or might be noise, and nothing traveling on the bus says which.

Every student project ends up inventing its own private theory of the room just to decide whether to pay attention to the next event. Most of those theories are incomplete, and none of them agree with each other, because each one is built from a partial view.

The orchestrator's first job — before anything clever — is to hold context once, on behalf of the whole room, and make sure every subscriber gets the same answer to *"what's happening right now?"* Once they do, most of the routing problem stops being a problem.

---

## What This Means for the White Paper

The thesis isn't "we built some cameras and a bot." It's: **we built a physical-world multi-agent system where cameras, student code, and human participants coordinate through a shared event bus — and here's what we learned about how a room builds shared context, how that context has to travel to the agents that need it, and what breaks when it doesn't.**

That's a genuinely novel contribution. Very few multi-agent papers deal with mixed human/software/sensor populations operating in real time in a shared physical space. The classroom isn't a demo of the idea — it's a semester-long experiment in it.

---

## White Paper Structure

### Abstract
Brief summary of the problem (coordination in mixed human/software/sensor environments), the approach (a classroom-scale multi-agent system built over one semester), and the key findings.

### 1. Introduction
Why this matters. The gap in current multi-agent research — everything assumes software-only agents. Frame the classroom as a research environment, not just a teaching tool.

### 2. Related Work
Multi-agent systems literature (AutoGen, CrewAI, LangGraph, MCP). Smart environments and ambient intelligence research. Human-in-the-loop systems. Where this work sits relative to each.

### 3. System Architecture
The agent taxonomy (perception, reactive, human). The bus (classroom API, SSE, event log). How the pieces were built and in what order. The design decisions and why they were made that way.

### 4. Agent Descriptions
Each agent type in detail — what it perceives, what it emits, how it coordinates. Camera-detectors, Discord bot, student projects, V-JEPA probe, humans. Include the table from this document.

### 5. Coordination Patterns
What worked with flat/reactive coordination. Where it started to strain. The routing problem — ambient vs. explicit vs. directed. The whiteboard sessions and how the orchestrator design emerged from observing the system under load.

### 6. The Orchestrator Design
Context state — what the room is doing right now, across temporal, social, task, and historical dimensions. Context-aware routing policy — which signals reach which subscribers, at what salience, given the current context. Pushing context to specialist agents so they don't have to guess. What's been designed, what's been built, what remains.

### 7. Observations and Findings
What we learned from running this for a semester. Context as the first-order problem — most coordination failures trace back to agents operating without a shared situational frame. The data-without-context failure mode: technically true readings that are practically meaningless because no one has told the downstream agents what the room is doing. How students naturally reasoned about their projects as agents. What surprised us.

### 8. Discussion
Implications for multi-agent system design beyond this classroom. What changes when agents include people and physical sensors. Limitations of this work. What we'd do differently.

### 9. Future Work
The orchestrator buildout. Cross-detector fusion. Depth features. Multi-camera coordination. Scaling beyond one room.

### 10. Conclusion
Restate the thesis and the contribution.

### References
