> **Working draft.** This document is being actively revised. It is linked early from the [companion story site](https://smartobjects.design) and released here in draft form to support feedback rather than to represent a finished argument. A polished version is expected on a 3–6 month horizon. Substantive feedback welcome via [GitHub issues](https://github.com/kandizzy/smart-objects-cameras/issues); for private or longer-form comments, `ckengle at sva dot edu`. Last updated **2026-05-06**.

# The Classroom as a Multi-Agent System

This system was always designed around a single premise: cameras should be conversational agents, not passive sensors. You talk to them, they talk back, they talk to each other. The classroom isn't monitored — it's inhabited by software that participates. That's a multi-agent system, even if we never used the term.

*See [`scoping-notes.md`](scoping-notes.md) for the early-semester thinking that became the structure of this paper.*

---

## Abstract

This paper describes a federated multi-agent system built at an interaction-design school over a single semester. The classroom puts three kinds of agent on a shared bus: perception agents (cameras and a V-JEPA classroom-state probe), reactive agents (eleven student projects spanning sensing, reaction, and physical actuation), and human agents (the students and instructors physically present). The bus contract — a single canonical event shape, three salience values, per-project consume/emit contracts — is small by design, so the federation's onboarding cost stays below the pleasure of the artifact. The cross-project conductor that would route events by context across the agent population is designed but not built; live integration of all eleven student agents onto the bus is the next iteration's work. The contribution is the architecture and the design discipline together: what multi-agent systems work looks like when humans are agents and the school producing it is a design school.

## 1. Introduction

Most people's smart object of choice is their smartphone. The class started a semester ago with a question about why. Given that and given the analysis we did of other products on the market that didn't become widely adopted, we decided to shift our efforts to making the space around us a smart object. With that, we decided to make a smart classroom.

One interesting challenge of making a smart object that will be used by multiple people is that it goes against the smartphone paradigm, which is normally just you and your smartphone interacting. The other challenge is the concept of surveillance: if something is smart, it's either listening to you or looking at you. It can feel unsettling. Who does the data belong to? Should opting in — or out — be the default? Should you always have the ability to opt in or out of such surveillance?

With that framing, the students embarked on their own ideas of what would be helpful in a classroom. In the end, we tried to weave them all together to see how those different modes of functionality coalesced into one intelligent system.

Unlike most, our multi-agent system included people. We had perception agents, like cameras that sense the room. We had reactive agents, like student projects that responded to what cameras sensed or to other activity in the room. And we centered ourselves as agents — humans on the same bus, asking where we fit and how to keep our agency.

## 2. Related Work

This system sits at the intersection of several research traditions, each of which solved part of the problem we encountered. None of them solved all of it. What follows traces the lineage, names the gaps, and positions the classroom's contribution against each.

*A note on sources: several of the works cited here are behind publisher paywalls. Where full texts were not accessible, we cite published abstracts and widely available descriptions of these systems. All quotations are drawn from abstracts or publicly available materials unless otherwise noted.*

### Ubiquitous Computing and Calm Technology

Mark Weiser argued in 1991 that the most profound technologies "weave themselves into the fabric of everyday life until they are indistinguishable from it" [1]. The smart classroom takes this literally: cameras, event buses, and routing policies are infrastructure that recedes into the room so the room can act on behalf of the people inside it. Weiser and Brown's later work on calm technology [2] — computing that "informs but doesn't demand our focus" — describes exactly what context-aware routing is trying to achieve. A fatigue alert during a lecture should reach the instructor quietly, not interrupt the room. The event should move from periphery to center only when the context warrants it.

Cook, Augusto, and Jakkula's survey of ambient intelligence [3] formalized this into a three-layer architecture: sensing, reasoning, and acting. The classroom maps cleanly onto it — detector agents sense, the orchestrator reasons about context, and student projects act — but with a difference the AmI literature did not anticipate. In the classroom, the "acting" layer is not a fixed set of actuators. It is a population of student-built agents that changes every semester, each with its own goals, and the system has to coordinate them without knowing in advance what they will be.

### Smart Rooms and Instrumented Spaces

The earliest smart classroom research, Georgia Tech's Classroom 2000 [4], captured lecture audio, video, and whiteboard strokes for later student review. It was a landmark project, but it was fundamentally passive — a recording system, not a participant. The cameras observed; they did not act.

Stanford's iRoom [6] and the Aware Home [5] moved closer to what we are building. iRoom coordinated multiple displays and input devices in a meeting room through middleware, treating the room as a collaborative workspace rather than a container. The Aware Home embedded sensing throughout a domestic space and treated it as a living laboratory. Both demonstrated that rooms could be programmable. But both coordinated *devices* — screens, sensors, appliances — not *agents* with independent goals and state.

The closest architectural predecessor is Gaia [7], a middleware platform from UIUC that treated physical spaces as first-class computational entities with their own operating system. Gaia's "active space" abstraction — a room with discoverable services, event channels, and context management — is structurally similar to the classroom's FastAPI event bus with SSE subscriptions and agent registration. The critical difference is population. Gaia coordinated homogeneous device services. The classroom coordinates cameras, student software, physical objects, and humans — a heterogeneous agent population where some agents reason, some just sense, and some are biological.

Streitz's Roomware and i-LAND project [8] pioneered the idea that furniture and architectural elements could themselves be computational agents — interactive walls, tables, and chairs that participated in the room's behavior. The student projects in this classroom (projected environments, rovers, physical feedback objects) are contemporary Roomware: physical things that sense, compute, and publish events to a shared bus.

### Tangible and Embodied Interaction

Ishii and Ullmer's "Tangible Bits" [9] proposed giving physical form to digital information, coupling the worlds of bits and atoms through tangible user interfaces. The student projects in this classroom are TUIs by Ishii's definition — a squeeze ball that publishes stress events, a rover that streams a dog's presence, a projected forest that responds to the room's energy. Each couples a physical artifact to a computational process via the event bus.

Dourish's *Where the Action Is* [10] provides the theoretical grounding for why this matters. Dourish argues, drawing on phenomenology, that meaning arises from engaged practice in the world, not from abstract symbolic manipulation. A classroom where students build physical agents that participate in a shared computational environment is embodied interaction in Dourish's sense — the learning happens through the construction and situated use of the artifacts, not through reading about multi-agent systems.

Krueger's Responsive Environments and VIDEOPLACE [11], dating to the late 1960s, demonstrated the fundamental loop decades before any of this vocabulary existed: cameras perceive human bodies in a room, and projectors respond in real time. The OAK-D detectors running person detection and gaze tracking are doing what Krueger did, with better models and a coordination layer he never needed because he was working with a single camera and a single response surface.

### Dynamicland

Dynamicland [12], Bret Victor's building-scale communal computer in Oakland, is the most direct inspiration for this classroom and the closest existing system. In Dynamicland, ceiling-mounted cameras and projectors recognize ordinary physical objects — index cards, books, 3D prints — and illuminate them with computational behavior. Programs are physical objects. The entire system runs on a single tightly integrated runtime called Realtalk, and all development happens within Realtalk itself, not on laptops.

The parallels are obvious: cameras perceive the room, physical objects are computational agents, multiple people collaborate simultaneously in a shared space. Both systems treat the room, not the screen, as the primary computational surface.

The differences are architectural and pedagogical. Dynamicland runs on a monolithic runtime — every object is a Realtalk program, and joining the system means writing Realtalk code. The classroom uses a federated architecture: new agents join by connecting to the event bus over HTTP, in whatever language and framework the student prefers. A p5.js sketch in a browser, a Python script on a Raspberry Pi, and a Discord bot are all first-class agents on the same bus. This federation is not a compromise — it is the design. The classroom is a pedagogical environment where students build the agents themselves, and they need to be able to build in whatever they already know.

Dynamicland is a research laboratory where a small team builds a unified vision. The classroom is a teaching environment where eleven students each build a piece of a system that has to cohere without anyone having designed the whole thing in advance. The event bus and the orchestrator exist precisely because there is no Realtalk — no single runtime that holds everything together.

### Multi-Agent Foundations

Minsky's *Society of Mind* [13] proposed that intelligence emerges from vast numbers of individually simple agents organized into teams and hierarchies. The classroom is a modest instantiation of this: camera agents, student project agents, the orchestrator, and humans — each limited on its own, collectively producing behavior that looks like a room that understands what it is doing.

Wooldridge and Jennings [14] defined the canonical agent properties: autonomy, reactivity, proactivity, and social capability. The classroom distributes all four across a heterogeneous population. Camera agents are reactive and social (they detect and publish). The orchestrator is proactive (it reasons about phase and routes accordingly). Student projects vary — some are purely reactive, some incorporate LLMs and plan. Humans are autonomous in ways the system cannot model. No single agent has all four properties. The system as a whole does.

Brooks' subsumption architecture [15] — intelligent behavior emerging from layered reactive behaviors without world models — describes the detector agents accurately. They sense and respond without maintaining models of the room. The orchestrator adds a deliberative layer on top, creating the hybrid reactive-deliberative architecture that the robotics community argued about in the 1990s, implemented here with web infrastructure instead of robot controllers.

The event bus itself is a blackboard system in the sense of Hayes-Roth [16]: a shared information space where independent knowledge sources communicate indirectly, monitoring the blackboard and contributing when their expertise is relevant. The addition of context-aware routing — where the orchestrator decides which events are salient given the current state — extends the classic blackboard with content-based filtering that early blackboard systems lacked.

### LLM Multi-Agent Frameworks

The current wave of multi-agent systems — AutoGen [17], CrewAI [18], LangGraph [19], applications built on MCP [20] and the A2A protocol [21] — assumes all agents are software processes, typically LLM-backed, communicating through function calls, tool use, or structured message passing. These frameworks have advanced the field rapidly, particularly in demonstrating that complex tasks can be decomposed across specialist agents with a supervisor coordinating.

CrewAI's supervisor-specialist pattern closely resembles the classroom's orchestrator-detector architecture. LangGraph's state-machine approach to agent workflow mirrors the orchestrator's phase logic. MCP and A2A address the same interoperability problem the event bus solves: how do heterogeneous agents discover each other and coordinate?

But all of these operate entirely in the digital domain. Their agents are processes. Their "environment" is a conversation thread or a shared memory buffer. They have no cameras, no physical objects, no humans on the bus. The classroom's contribution is not a better agent framework — it is evidence that the same coordination patterns work when the agent population includes things that sense physical reality, things built by students who had never heard the term "multi-agent system," and people sitting in chairs.

### Context-Aware Computing

Schilit, Adams, and Want [22] defined context-aware computing in 1994 as software that "adapts according to the location of use, the collection of nearby people, hosts, and accessible devices." Dey and Abowd [23] broadened the definition: context is "any information that can be used to characterize the situation of an entity," encompassing identity, activity, time, and location. The orchestrator's context state — temporal, social, task, and historical dimensions — is a direct implementation of Dey's framework, applied to event routing rather than application adaptation.

The routing mechanism itself is a form of content-based publish/subscribe [24], where subscribers receive events based on the event's content and the subscriber's declared capabilities, not just a channel name. Traditional topic-based pub/sub (subscribe to "fatigue" and get every fatigue event) is too coarse for a room where the same event type means different things in different contexts. Content-based filtering, modulated by the orchestrator's context state, is what makes the routing context-aware rather than merely categorical.

### Smart Classrooms with Computer Vision

Recent surveys of AI in smart classrooms [26] document a growing ecosystem of camera-based sensing: engagement detection, attention monitoring, emotion recognition. Kim's work on ambient intelligence in classrooms [27] used ambient sensors to detect student engagement and adapt learning strategies. Canedo and Trifan [28] tracked gaze direction and facial expressions to infer attentiveness.

These systems are technically similar to the classroom's detector agents — the same models, the same signals. The architectural difference is that these systems treat cameras as data sources feeding a centralized controller. In the classroom, cameras are agents on a bus. They publish events. Other agents subscribe. The orchestrator routes. No central controller owns the pipeline end-to-end. This decentralization is not an academic preference — it is a practical requirement. When eleven student projects need to consume camera data in eleven different ways, a centralized pipeline cannot serve them all. The bus can.

### Constructionism

Papert's constructionism [29] — the theory that learning is most effective when people build tangible things in the real world — is the pedagogical backbone. The students did not study multi-agent systems and then implement one. They each built a project (a timer, a grammar coach, a projected forest, a virtual dog) and discovered, over the course of a semester, that their projects were agents on a shared bus, that the bus needed context to route effectively, and that an orchestrator was the missing piece. The multi-agent framing emerged from the practice, not the other way around. The event bus is the medium through which student-constructed artifacts become part of the room's intelligence.

### Positioning

No prior work combines all of the following: a federated event bus (not monolithic like Dynamicland), a mixed agent population of sensors, software, and humans (not all-software like AutoGen), context-aware routing that modulates event salience by situational frame (not flat pub/sub), and student-constructed agents as first-class participants whose design was not known in advance. The closest single ancestor is Gaia, updated for an era of LLM-backed orchestration and heterogeneous agent populations. The classroom is not a framework — it is a semester-long experiment in what happens when you build one of these systems with real cameras, real students, and real coordination problems.

## 3. System Architecture

The smart classroom is a *federated* multi-agent system: each agent is independently owned and implemented — running in its own language and on its own hardware — joining the system by speaking a shared protocol rather than living inside a shared runtime. The agent population spans software, hardware, and human, all coordinating through a single bus. This section names the agent taxonomy, describes the bus, and traces the build order that produced both.

### 3.1 Three Classes of Agent

Most multi-agent work right now — AutoGen, CrewAI, LangGraph, MCP-based systems — assumes all agents are software talking to software. This classroom puts three fundamentally different kinds of agent on the same bus:

1. **Perception agents** — cameras that sense physical reality. No language, no reasoning, just continuous signal at frame rate.
2. **Reactive agents** — student projects that respond to room events. Some are LLM-backed, some aren't. A p5.js sketch that changes color when `room_mode` flips is still an agent by any meaningful definition.
3. **Human agents** — the students and instructors physically in the room, whose goals the system exists to serve.

This mixing is the point. The design question was never "how do we get software agents to collaborate" — it was "how do we build a coordination layer where cameras, code, and people coexist as participants in the same space." That's a different and harder problem than what current multi-agent frameworks are solving, and it's why the system looks the way it does.

The full inventory of agents at the close of the semester:

| Agent | Class | Loop | State | Goal | Channel |
|-------|-------|------|-------|------|---------|
| **Camera-detectors** (person, fatigue, gaze, whiteboard) | Perception | 5–30 FPS | Local status files + API state | Report what I see | `POST /push/state` |
| **V-JEPA probe** | Perception | Classification @ 0.1 Hz | Prediction history | Label what's happening | `predicted_class` field via push |
| **Discord bot** | Reactive | Command event loop | Reads API state | Answer humans when asked | Discord commands + webhook |
| **Student projects** (Tony, Forest, Timer, Inprint, Gus, NodCheck, …) | Reactive | SSE subscription loop | Project-local events | React to the room in a project-specific way | `GET /subscribe/state`, `POST /projects/:id/events` |
| **Smart Stage** | Reactive (room-state orchestrator) | Polling + scheduling | Room phase, ambience state | Conduct ambience across phases | Polls V-JEPA, writes `/mode` |
| **Humans in the room** (instructors, students) | Human | Biological | Memory, intent | Varies per person | Physical presence + Discord |
| **Cross-project conductor** (designed, not built) | Reactive (cross-project orchestrator) | Context state machine | Shared room context | Route by context across all agents | Would own `/mode` as the writable endpoint |

At semester end, the spine of the system — camera-detectors, the overhead dashboard, the V-JEPA probe, the Discord bot, and Smart Stage — was deployed and running on the live bus. The eleven student projects existed as standalone demos; their integration with the API bus was prototyped in the simulator (see §5) but did not run end-to-end with all student agents live in the classroom. The cross-project conductor exists in design only. Section 4 describes each entry in detail; this paper is honest about the gap between architecture-as-designed and architecture-as-deployed.

### 3.2 The Bus

The classroom API functions as a **message bus with a shared memory model**. Three primitives carry the system:

- **Server-Sent Events (SSE)** — the publish/subscribe channel. Any agent connects to `GET /subscribe/state` and receives a live stream of room state changes.
- **The `classroom_events` table** — the event log. An append-only record of everything that has happened on the bus, queryable for context reconstruction and post-hoc analysis.
- **The `student_projects` registry** — the agent directory. New projects register themselves and become discoverable to other agents.

These were the first things we built, because the system needed a way for independent actors to find each other and share state before any coordination was possible.

### 3.3 Build Order

The semester prioritized the agents and the bus first. You can't design coordination until you know what's being coordinated, so the order was deliberate:

1. **Cameras + detector scripts** — the physical input layer. OAK-D-POE devices on three Raspberry Pi 5s, with detector scripts (person, fatigue, gaze, whiteboard) running on each.
2. **Discord as the first bus.** Before the FastAPI layer existed, Discord served as the bus. Public bots — `orbit`, `gravity`, `horizon`, one per Pi — published detector state to a shared channel where the class could see and address it. Each student also ran a personal bot for direct messages, so they could develop against a private Discord surface without spamming the class. This freed the team from building the software bus before agents could be wired up at all: students experienced what talking to a camera felt like weeks before any FastAPI code existed.
3. **The classroom API** — FastAPI + Postgres + SSE. Took over from Discord as the structured bus that let agents find each other, push state, and subscribe to events programmatically. Discord stayed in the system, but its role narrowed to the conversational surface described in §4.2.
4. **Student projects** — eleven projects, each designed as its own agent on the federation pattern. Variety was the point: some LLM-backed, some not, some physical, some browser-based. Built and demonstrated as standalone projects during the semester; integration with the API bus was prototyped in the simulator (§5) but did not run live with all student agents at once.
5. **Smart Stage** — a room-state orchestrator that polls V-JEPA, reads the schedule, and adjusts ambience accordingly. Roughly 990 lines of Python on a Raspberry Pi 5.
6. **Cross-project conductor** — designed, scoped, prototyped in the simulator, not built into the live system. The remaining work after one semester.

Building agents and bus first was a deliberate constraint. With the layer that knows the room — the conductor — deferred to last, every other agent had to operate without a shared situational frame. That gap, observed across eleven projects over a full semester, is what motivates the coordination patterns analyzed in §5 and the orchestrator design proposed in §6.

## 4. Agent Descriptions

Each agent in §3 has its own loop, its own state, and its own channel into the bus. This section describes what each one perceives, what it emits, and how it coordinates with the rest. The companion-site narrative covers the design philosophy of the student projects in detail; the focus here is architectural — what each agent contributes to the room as a multi-agent system.

### 4.1 Perception Agents

**Camera-detectors.** Four detector scripts run across three Raspberry Pi 5s (named `orbit`, `gravity`, `horizon`), each paired with an OAK-D-POE camera and one or more on-device neural networks. Each detector observes one signal:

- **`person_detector`** — YOLOv6-nano on COCO class 0. Publishes presence, count, and per-detection bounding boxes.
- **`fatigue_detector`** — YuNet face detection → MediaPipe FaceLandmarker → eye aspect ratio (EAR) and head-pose Euler angles via `solvePnP`. Publishes a binary fatigue state with confidence.
- **`gaze_detector`** — Three-stage pipeline: YuNet face detection → head-pose estimation → ADAS gaze estimation. Publishes a unit gaze vector and a discretized direction (`board`, `phone`, `nothing`).
- **`whiteboard_reader`** — PaddlePaddle text detection → text recognition. Publishes detected text regions, recognized strings, and confidence per region.

All detectors apply temporal debouncing (1.5–2 seconds) before emitting state changes, so a flicker doesn't propagate as an event. State writes go to local JSON status files (read by the Discord bot) and to the classroom API via `POST /push/state` (read by everything else). Each publishes at frame rate (5–30 FPS) but emits onto the bus only on confirmed transitions.

**Overhead dashboard.** Gordon Cheng's stitched bird's-eye view of the room. Three ceiling-mounted PoE cameras feed into a single canvas, calibrated against blue ArUco markers anchored to the table (Fig. 01). YOLO6-nano runs on the stitched canvas for person detection, producing a flat plan of the room with dots representing detected bodies. The dashboard is a perception agent that emits aggregate occupancy and approximate position rather than per-camera detections.

**V-JEPA probe.** A custom-trained classification head on top of Meta's V-JEPA video-pretrained backbone, hosted on a GPU machine across the network. Polled by Smart Stage every ten seconds with a short clip from the overhead cameras; returns a class label drawn from the trained set: `empty`, `lecture`, `group_work`, `break`. Unlike the detectors, V-JEPA does not push to the bus on its own initiative — Smart Stage queries it. The probe was trained first at home on Carrie alone, then in a single classroom session with the students using the procedure documented in `VJEPA_SETUP.md` and `CLASSROOM_CLIP_STRATEGY.md`. It is the closest the system comes to a foundation model at the perception layer.

### 4.2 Reactive Agents

**Discord bot.** Originally the system's bus (see §3.3), Discord now functions as the room's conversational surface. A long-running Python process listens on a Discord channel. It reads the JSON status files written by the local detectors and answers commands (`!status`, `!detect`, `!screenshot`, `!whiteboard`) by quoting current state back to the channel. Per-camera personality (separate bot tokens for `orbit`, `gravity`, `horizon`) gives the room a distributed conversational surface — a student can ask any of three cameras independently. The bot also subscribes to detector state and emits unsolicited Discord messages on important transitions (e.g., a fatigue alert routed as a DM in opt-in mode) when configured to do so.

**Smart Stage.** The room-state orchestrator. Approximately 990 lines of Python on a Raspberry Pi 5. Smart Stage holds a single state machine with named phases (`pre-arrival`, `arrival`, `lecture`, `break`, `group-work`, `labs-and-demos`, `wrap`) and runs three loops:

- A **schedule loop** that reads the day's session plan and fires phase transition cues at known times.
- A **V-JEPA polling loop** that queries the probe every ten seconds and updates phase belief based on the predicted class.
- An **ambience loop** that adjusts music volume, captions on/off, and projector content based on the current phase.

Smart Stage is the closest thing to a conductor that exists in the live system — but it conducts ambience, not cross-project routing. That distinction is what motivates §6.

**Student projects.** Eleven student projects were built during the semester, each designed to subscribe to `GET /subscribe/state`, filter for events relevant to its concern, and emit its own events via `POST /projects/:id/events`. By semester end, each existed as a standalone demo; the live federation — multiple student agents on the bus simultaneously — was prototyped in the simulator (see §5) but did not run end-to-end in the classroom. The projects span sensing, reaction, and physical actuation:

- *Sensing.* **Sleep Detection** (Kevin Shi & Mingyue Zhou) — face landmarks → EAR → opt-in drowsiness alert routed laterally to a chosen peer, not to the instructor. **Focus Beam** (Feifey Wang) — MediaPipe gesture detection → projected slide-region dimming. **Assignment Tracker** (Shuyang Tian) — captions + whiteboard OCR → deadline and task extraction. **NodCheck** (Kathy Choi) — head-pose nod/shake detection during a comprehension window. **Forest in the Classroom** (Sophie Lee) — voice-energy detection → generative trees on a back-wall projection.
- *Reaction.* **Tony** (Ramon Naula & Shuyang Tian) — an 18-channel robotic spider with YOLO running behind its eyes and Groq powering its words; present in both the classroom and on Discord. **Inprint** (Darren Chia) — handwriting capture from any surface, surface-agnostic by design. **English Communication Coach / Lumi** (Yuxuan Chen) — private, delayed, gentle feedback after class. **Gus Mode** (JuJu Kim, Kathy Choi, Seren Kim) — a Viam Rover at a remote home streaming a segmented dog onto the classroom wall.
- *Actuation.* **Timer** (Phil Cote) — arUco tag detection on a foam-core board → countdown displayed on a shared screen, set by tactile gesture at the whiteboard.

The companion-site narrative describes each project in design and pedagogical detail. From an architectural standpoint, the variation among them — different languages, different hardware, different goals, different temporal cadences — is what the bus + registry pattern was designed to support. Whether it does, with all eleven agents live on the bus at once, is the next iteration's experiment.

### 4.3 Human Agents

The instructors and students physically present in the room are first-class agents in this system. They have biological perception loops the system cannot directly query, intent and memory the system cannot model, and goals the system exists to serve. Their channels into the bus are physical presence (observable by the cameras as posture, motion, count) and Discord (observable as text by the bot and by every project that subscribes). Treating them as agents rather than as users — co-equal participants on the same bus rather than recipients of services — is a design choice; §5 and §8 examine its consequences.

### 4.4 The Cross-Project Conductor (Designed, Not Built)

The final agent in the inventory exists in design but not in the live system. The conductor would sit above all the others as a context-aware router: it would hold a unified context state (temporal, social, task, historical), receive events from every perception and reactive agent on the bus, and dispatch them — at the right salience, to the right subset of subscribers — based on what the room is doing. Smart Stage already does this for ambience within its own scope; the conductor would do it across the whole agent population. §6 describes what the conductor actually needs to be. The reason it exists in design but not in deployment is the build-order constraint named in §3.3: agents and bus first, conductor last, when there is enough live behavior to design coordination against.

## 5. Coordination Patterns

### 5.1 What worked under flat coordination

The system's spine held through the semester. Cameras detected; Smart Stage orchestrated room state; the V-JEPA probe classified posture; the Discord layer carried coordination between people and projects. None of this required a conductor — flat publish/subscribe was enough.

What flat coordination gave the room, beyond raw function, was two things worth naming.

The first is *agency preserved*. Each smart object had a specific, named purpose — a timer, a fatigue alert, an OCR, a gaze reader, a generative projection. None were all-seeing surveillance devices. The room's intelligence was distributed across artifacts whose purposes the room's occupants understood and agreed to. That was preserved by design and held up across implementations because the federation pattern made each agent's job legible.

The second is *legibility*. Because Discord was the bus through which the early classroom communicated, the room's coordination was readable by humans. Open a channel and you could watch the room think. Most multi-agent systems hide their coordination inside protocols nobody reads at runtime; in this classroom, you could read it.

Phil's Timer is the canonical example. ArUco tags on a foam-core board face up; Horizon (an OAK-D on a tripod) reads them; Phil's display receives the detection event and starts a countdown on a shared screen. Detection → bus → state machine → display, with no orchestrator deciding which signal mattered. The whole loop ran on flat coordination, and it worked.

### 5.2 What we did not get to

What we did not get to, by semester end, was the live federation. None of the eleven student projects connected to the FastAPI bus during the semester. Each was built as a standalone demo, with the federation pattern as a design target — projects were structured to subscribe to bus events and emit their own — but the integration step, where eleven student-built agents act on each other's events through the live bus, did not happen.

In place of that live integration, the class built a *simulator*: a single page that animates the room across one Monday session, replaying a curated event stream through the architecture as if all eleven projects were running at once. The simulator's purpose is not validation — it cannot prove the federation works — but communication. It shows what coordination would look like in time, against what kinds of events, between which projects. It made the architecture visible to students who had not yet wired their projects to the bus.

The integration step is where the coordination story actually begins. We got to its threshold and stopped.

### 5.3 Where it strained

Across the semester, the strains on the system rhymed. Five patterns recurred often enough to name:

1. **Camera placement.** Many projects needed a camera positioned for a specific task — Sleep Detection on a student's face, NodCheck on a student during the comprehension window, Inprint over a writing surface, Timer on the foam-core board, the whiteboard OCR scripts on the whiteboard itself. Each project's success depended on the camera being in the right place at the right time. §5.4 returns to this — it is the dominant pattern.

2. **Surveillance trade-off.** Overhead cameras solve the placement question, but reintroduce the question §1 foreclosed. Sleep Detection felt this most acutely: do you watch students from above to know they are tired, when watching them is the thing they do not want to be watched by? The class's answer — lateral peer notification, opt-in only — is itself a coordination decision, a routing policy applied at the design level rather than at runtime.

3. **Infrastructure gaps.** Several projects required infrastructure beyond what the federated software could provide. Focus Beam needed addressable lighting. Forest needed projection plus microphones. Assignment Tracker needed a long-running language model separately scanning the captions for utterances that sound like assignments. These are not coordination failures in the strict sense; the bus was ready to carry their events. They are *dependency* failures: an agent in a federation can only coordinate over what it can produce or consume.

4. **Personal-device modality.** Some projects — the English Communication Coach is the clearest case — want to live on a personal device because the interaction is private. But the classroom encourages those devices to be put away. If the agent's natural home is a device the room is asking you not to use, where does the agent run? This is the dual of the camera-placement question: not "where is the camera" but "where is the receiver."

5. **The live-classroom gap.** Several projects had working demos that never ran in the live classroom. Forest is the canonical example — the projection responds to voices in a tabletop demo, but the room never saw it lit up during a real class. This is a runway problem rather than a design failure. It also names a concrete unknown: the system's behavior at scale, with all agents and all humans live, is not yet observed.

### 5.4 The "where's the camera?" pattern

The camera-placement pattern is not really a hardware problem. It is a coordination problem dressed as a logistics one. Multiple projects, with different purposes, needed cameras in different positions — but the room had three cameras, mounted on tripods, pointed where someone last left them. The coordination question is: how does the room help itself *know* where its cameras should be?

The answer, surfaced repeatedly through the semester, was that the camera should *tell* the user where it needs to be. Treat the camera as a conversational agent — same architecture as everything else, capable of speaking on Discord — and the placement question becomes a dialog: *"to use me for the timer, walk me to the whiteboard and set me down with my long axis along the board's bottom edge."* The user picks up the tripod and moves it. The camera confirms its new view and re-emits its readiness on the bus.

This is the same architecture the rest of the system uses, with a different actuator: the human is the actuator. On a tripod budget, that is what is available. Replace the human with a robot — Boston Dynamics' Spot, recently demonstrated using its gripper-mounted camera to read tasks from a whiteboard before acting on them — and the camera repositions itself. The architecture is independent of the embodiment. §6 holds the conductor design that would carry the policy ("which camera should be where, given what the room is currently doing"); §9 returns to the robotic embodiment.

This pattern is the strongest argument for why a cross-project conductor is the right next piece. It is where flat coordination meets its limit and where context-aware routing — the conductor's job — would do the most useful work.

### 5.5 What we reached

The semester ended at the start of coordination, not at its end. What we reached was a working perception layer, a working room-state orchestrator, a working conversational bus, eleven student-built projects designed for the federation, a simulator that shows what coordination would look like in time, and a design for the cross-project conductor that would hold the system together across all of them.

What is missing is the integration step — eleven agents on the bus simultaneously, acting on each other's events in a live room — and the conductor that would route between them. Both are scoped. Both are the next year's work, not this one's.

The most honest thing to say is the thing one of us said directly: *we reached a point where now we can start the project. Now we can design the interfaces between them. Now we can study what this would actually feel like.* That moment — when the question stops being "what should we build" and becomes "what should we build *next*" — is the outcome of the semester. The sections that follow describe what *next* means.

## 6. The Orchestrator Design

The orchestrator design has two layers. The first is already built: the bus contract — a single canonical event shape, three salience values, semantic categories, per-project consume/emit contracts — implemented in Bruno Kruse's classroom-heartbeat Node server (`classroom-api/`). The second is designed but not built: the *conductor*, a context state machine that would extend the bus contract with context-aware routing across the agent population. This section describes both, with explicit boundaries between what is running and what is scoped.

### 6.1 The bus contract (built)

The classroom-heartbeat server owns the bus contract. The contract is a single canonical event shape:

```json
{
  "id": "evt-...",
  "event_type": "sensor.light.changed",
  "category": "sensor",
  "source": "class-object-simulator",
  "target": null,
  "salience": "broadcast",
  "created_at": "2026-04-22T00:00:00.000Z",
  "payload": {}
}
```

Required fields are `id`, `event_type`, `source`, `created_at`, and `payload`. The server normalizes `category` (from a known prefix taxonomy — `sensor.*`, `detection.*`, `projection.*`, `surface.*`, `room.*`, and others) and `salience` based on the event type.

**Salience.** Three values, matching the routing taxonomy from §3: `ambient` (whispered on the bus for any subscriber to pick up), `broadcast` (surfaced to all interested subscribers as a notification), and `directed` (delivered to a specific target). The server applies a default salience rule based on event-type prefix — `debug.*` is ambient, `character.*` is directed, most others default to broadcast — but any agent can override by setting salience explicitly on the event.

**Project contracts.** Each student project registers with the server through a contract document declaring what events it `consumes` and what events it `emits`. The contracts are markdown files served at `GET /api/projects/{id}/contract.md`. They function as the agent directory: anyone joining the bus can ask the server what every other agent is listening for and producing. The contracts are the federation's coordination glue — without them, agents would have to assume the rest of the bus.

**Event ingest and propagation.** A single endpoint receives all events: `POST /api/action`. The server normalizes, validates, and re-emits via SSE on `GET /api/events`. Events are appended to a JSONL replay file and queryable via `GET /api/replay`. Project heartbeats and project-emitted events have their own paths (`POST /api/projects/{id}/heartbeat`, `POST /api/projects/{id}/events`) but flow through the same normalization.

**Compatibility layer.** Detector workers built before the heartbeat server can keep posting to the legacy `POST /push/state` endpoint; the server maps old detector fields to canonical dotted events (`person_count` → `class.presence.changed`, `predicted_class` → `classifier.probe.changed`, `gaze_direction` → `attention.direction.changed`, and so on). The bus accepts every agent on its own terms — federation in practice required this.

This layer is what the room runs on today.

### 6.2 What the conductor adds (designed)

The conductor sits above the bus contract. It does three things the bus does not.

**(a) A context state.** A unified representation of what the room is doing right now, across four dimensions:

- **Temporal** — what phase of the session we are in (lecture, break, group work, demos)
- **Social** — who is in the room, who is speaking, who is listening
- **Task** — what the room is collectively doing
- **Historical** — what just happened, and whether the current moment is a continuation or a shift

Today, every agent that needs context reconstructs it for itself by subscribing to whatever events seem relevant and guessing. The conductor would compute context once, in one place, and make it queryable.

**(b) A context-aware routing policy.** Which signals reach which subscribers, at what salience, given the current context. The bus contract gives us salience as a value on each event; the conductor decides what that salience should be based on the room's state. A fatigue alert during a forty-five-minute lecture should reach the instructor as a directed Discord DM; the same alert during a five-minute break should be ambient and ignored. Same event, different routing decision because the context changed.

**(c) A way to push context to specialist agents.** Agents that need to behave differently in different contexts should receive context updates rather than being asked to infer them from event traffic. The runtime-configuration mechanism that already exists — agents accept push updates to thresholds and parameters — is the seed of this. Generalize it to push the room's full context to any agent that registers interest, so each agent operates against the same situational frame as the rest.

These three pieces — context state, context-aware policy, and pushed context — are the difference between a federation that works at the level of individual agents and a federation that knows what room it is in.

### 6.3 The simplicity discipline

The classroom-heartbeat repository documents an engineering discipline that is itself part of the orchestrator design. From `engineering-rules.md`:

> Use direct function calls before abstractions. Use plain objects before classes. Use one file before a framework. Use one explicit endpoint before a generic routing system. Use one local JSON file before a configuration system. Use hand-written validation before adding a schema library.

The "do not add" list is more pointed: *"a capability router until multiple real providers exist; a project registry until the simulator proves it needs one; a config system with exactly one configuration."*

This is not just style. It is a design choice about what the orchestrator should and should not become. The temptation in a multi-agent paper is to scope a sprawling routing layer with capability descriptions, dynamic dispatch, and pluggable policies. The classroom-heartbeat approach refuses that until necessary. The conductor design in §6.2 is small on purpose: a context state, a routing policy, a push mechanism. Three things, not a framework. The discipline of the bus layer keeps the conductor honest.

What the orchestrator actually needs to be — given everything in §3 through §5 — is small. Holding context and using it to route. Nothing more, until the room's behavior tells us we need more.

## 7. Observations and Findings

### 7.1 The design-space dichotomy

There is a dichotomy in how people relate to technology. On one side, people don't want it at all — visible technology in a learning space reads as surveillance, as distraction, as someone else's design imposing itself on their attention. On the other side, when people do want it, they want it to be so seamless and advanced that they do not have to think about it at all — the calm-technology destination Weiser described.

The design space the classroom occupied was the gap between these poles. While the underlying capabilities have become increasingly magical, designing a useful and calm ambient system around them remains hard. *The cost is at the seams.* A smart room that almost works can feel worse than no smart room, because the rough edges are where attention is pulled away from whatever the room was trying to be in the first place.

The five strain patterns in §5 are what the gap looks like in practice. Camera placement is the partial-calm failure mode in its purest form: the camera works *if* you walk over and reposition it. Forest never running live is partial-calm at the room scale: the demo works at a desk; the projection never lit up during a class. Sleep Detection's surveillance worry is partial-calm at the social scale: overhead watching gives you the data, but reintroduces the thing the room was trying to avoid.

### 7.2 Working is a delight; more work than it saves is abandonment

When a project worked in the room, it was a real delight. Bruno setting the timer at the whiteboard and the team exclaiming *"Yes!"* — the smart classroom beginning to come to life — is the canonical example. What worked was specific, named, and useful, and the user could feel it was helping. That feeling is what we were aiming at across all eleven projects.

The inverse principle is also worth naming. Things that take more work than they save get left behind. Within this semester, that principle showed up not as students choosing the demo over integration but as integration being out of reach altogether. Eleven projects shipped as demos. The bus integration step — writing each project's contract document, registering with the heartbeat server, subscribing to events from other projects, debugging across the federated stack — was the next thing the work asked for, but a class of design students running on a single semester does not get to it. That isn't a failure of the students; it is the actual scope of an interaction-design class doing engineering work at the edge of its training.

The federation has an onboarding cost. For a federation to compose, that cost has to be lower than the pleasure of the artifact already in front of you. This is part of why §6.3's simplicity discipline matters. *Use one explicit endpoint before a generic routing system. Use one local JSON file before a configuration system.* The bus contract is small precisely so that the work of joining it is small. The next iteration's job is to make joining the bus light enough that design students with the time available can reach integration before the semester ends.

## 8. Discussion

Three implications follow from this work for multi-agent system design beyond the classroom.

First, the architecture must be small enough that joining is cheap. The classroom-heartbeat bus contract — single canonical event shape, three salience values, per-project consume/emit contracts — is small precisely so design students can join it without writing their projects around someone else's framework. A larger orchestration framework would have moved faster on engineering depth and slower on student ownership, and student ownership is where the multi-agent claim actually gets tested.

Second, context matters more than capability. Most coordination work in this classroom traced not to insufficient detector accuracy or insufficient agent intelligence but to agents operating without a shared situational frame. A fatigue alert is signal during a lecture and noise during a break; the same data, routed differently, has opposite meaning. The conductor design proposed in §6 is the response to this finding, but the finding stands independent of the design: in a room with mixed-population agents, the work of routing-by-context dominates the work of inference.

Third, the school shapes the contribution. This system was built at an interaction-design program, with design students doing engineering work at the edge of their training. The result is honestly a design-school multi-agent system: strong on the felt experience of the artifacts, the surveillance ethics, the calm-tech intent, and the agent-as-conversational-object framing — and limited at the integration layer, where engineering depth would have moved faster. A future iteration paired with a robotics school or company would unlock the embodiment threads this paper opens but does not close; a pairing with an engineering program would close the integration gap. What either pairing would lose is the design discipline that put humans on the bus in the first place. The design-school version is not a deficiency to be corrected; it is the form of multi-agent systems work this kind of school produces, and it is its own contribution.

**Limitations.** A single semester, an under-integrated federation, and a single classroom. The conductor design has not yet been observed under load. Each agent type is represented by a small number of student projects. None of the findings have been confirmed across rooms, across cohorts, or against engineering-school baselines. We expect the dichotomy and the federation-onboarding-cost finding to generalize; we have not yet tested whether they do.

## 9. Future Work

Seven directions for the next iteration of this work.

1. **The integration step.** Wiring the eleven student projects onto the live FastAPI bus, so the federation runs end-to-end. This is the work the semester's runway did not allow.

2. **The cross-project conductor.** Implementing the context state machine described in §6.2: temporal/social/task/historical context, context-aware routing, pushed context for specialist agents.

3. **Project inheritance.** The federation pattern is also an inheritance pattern. Phil Cote has expressed the hope that his Timer would be picked up by a future student. Verifying this — pulling Phil's repository, plugging into next year's bus, and confirming the timer runs without Phil running it — is a concrete test of whether projects survive their authors when the bus is the only thing they need to talk to.

4. **The single-occupant bus.** The author intends to run the integration over the summer at home, on a single-author bus, before the next cohort arrives to populate it. This is also a runway-management strategy: bring the bus past first integration before introducing new agents to it.

5. **Robotics partnership.** A pairing with a robotics company or robotics school would unlock the embodiment threads this paper opens but does not close — the conversational camera that walks itself, Tony as a fully-actuated multi-channel agent, physical actuators on the bus generally. Boston Dynamics' demonstration of Spot reading whiteboard tasks via its gripper-mounted camera is the existence proof; the architecture is ready for the embodiment to catch up.

6. **Cross-detector fusion, depth features, multi-camera coordination.** The OAK-D-POE hardware supports stereo depth and per-camera spatial reasoning that this iteration did not exploit. Fusion across detectors (gaze + fatigue + posture) and across cameras (the overhead dashboard with depth) is straightforward implementation work that would expand what the perception layer reports.

7. **Scaling beyond one room.** Nothing in the federation pattern requires a single room. The same bus contract with multiple rooms registering as agents — a multi-room conductor coordinating between them — is the path to a building-scale or campus-scale version of the same architecture.

## 10. Conclusion

The classroom this paper describes is a federated multi-agent system that includes people. By semester end, the spine — perception layer, room-state orchestrator, conversational bus — was running. Eleven student projects were built as standalone demos with the federation pattern in mind. The cross-project conductor and the live integration of all agents on the bus were designed and prototyped but not deployed.

The contribution is the architecture and the design philosophy together: a small bus contract that design students can join without writing into someone else's framework; salience-routing that distinguishes ambient, broadcast, and directed events; a conductor design that holds context once for the room rather than asking every agent to reconstruct it; and a deliberate engineering discipline — *use one explicit endpoint before a generic routing system* — that keeps the federation small enough to compose. None of this is a new framework; it is what multi-agent systems work looks like when humans are agents and the school producing it is a design school.

The system is not done. We reached a point where now we can start the project. The next iteration is what this paper was the ground for.

## References

See [`references.md`](references.md) for the full bibliography.

---

## Changelog

- **2026-05-06** — Initial draft. §1, §3 (with corrections from author), and §7.1–§7.2 (with corrections from author) are author-led; §2 inlined from `related-work.md` (author-authored); §6.1, §6.3 draw on Bruno Kruse's `classroom-api/docs/` (engineering-rules, event-contract); §4, §5, §6.2, §8, §9, §10, and the Abstract drafted with Claude assistance and reviewed by the author. Released as draft for invested-reader feedback.
