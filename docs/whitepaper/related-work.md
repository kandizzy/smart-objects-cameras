# Related Work

This system sits at the intersection of several research traditions, each of which solved part of the problem we encountered. None of them solved all of it. What follows traces the lineage, names the gaps, and positions the classroom's contribution against each.

*A note on sources: several of the works cited here are behind publisher paywalls. Where full texts were not accessible, we cite published abstracts and widely available descriptions of these systems. All quotations are drawn from abstracts or publicly available materials unless otherwise noted.*

## Ubiquitous Computing and Calm Technology

Mark Weiser argued in 1991 that the most profound technologies "weave themselves into the fabric of everyday life until they are indistinguishable from it" [1]. The smart classroom takes this literally: cameras, event buses, and routing policies are infrastructure that recedes into the room so the room can act on behalf of the people inside it. Weiser and Brown's later work on calm technology [2] — computing that "informs but doesn't demand our focus" — describes exactly what context-aware routing is trying to achieve. A fatigue alert during a lecture should reach the instructor quietly, not interrupt the room. The event should move from periphery to center only when the context warrants it.

Cook, Augusto, and Jakkula's survey of ambient intelligence [3] formalized this into a three-layer architecture: sensing, reasoning, and acting. The classroom maps cleanly onto it — detector agents sense, the orchestrator reasons about context, and student projects act — but with a difference the AmI literature did not anticipate. In the classroom, the "acting" layer is not a fixed set of actuators. It is a population of student-built agents that changes every semester, each with its own goals, and the system has to coordinate them without knowing in advance what they will be.

## Smart Rooms and Instrumented Spaces

The earliest smart classroom research, Georgia Tech's Classroom 2000 [4], captured lecture audio, video, and whiteboard strokes for later student review. It was a landmark project, but it was fundamentally passive — a recording system, not a participant. The cameras observed; they did not act.

Stanford's iRoom [6] and the Aware Home [5] moved closer to what we are building. iRoom coordinated multiple displays and input devices in a meeting room through middleware, treating the room as a collaborative workspace rather than a container. The Aware Home embedded sensing throughout a domestic space and treated it as a living laboratory. Both demonstrated that rooms could be programmable. But both coordinated *devices* — screens, sensors, appliances — not *agents* with independent goals and state.

The closest architectural predecessor is Gaia [7], a middleware platform from UIUC that treated physical spaces as first-class computational entities with their own operating system. Gaia's "active space" abstraction — a room with discoverable services, event channels, and context management — is structurally similar to the classroom's FastAPI event bus with SSE subscriptions and agent registration. The critical difference is population. Gaia coordinated homogeneous device services. The classroom coordinates cameras, student software, physical objects, and humans — a heterogeneous agent population where some agents reason, some just sense, and some are biological.

Streitz's Roomware and i-LAND project [8] pioneered the idea that furniture and architectural elements could themselves be computational agents — interactive walls, tables, and chairs that participated in the room's behavior. The student projects in this classroom (projected environments, rovers, physical feedback objects) are contemporary Roomware: physical things that sense, compute, and publish events to a shared bus.

## Tangible and Embodied Interaction

Ishii and Ullmer's "Tangible Bits" [9] proposed giving physical form to digital information, coupling the worlds of bits and atoms through tangible user interfaces. The student projects in this classroom are TUIs by Ishii's definition — a squeeze ball that publishes stress events, a rover that streams a dog's presence, a projected forest that responds to the room's energy. Each couples a physical artifact to a computational process via the event bus.

Dourish's *Where the Action Is* [10] provides the theoretical grounding for why this matters. Dourish argues, drawing on phenomenology, that meaning arises from engaged practice in the world, not from abstract symbolic manipulation. A classroom where students build physical agents that participate in a shared computational environment is embodied interaction in Dourish's sense — the learning happens through the construction and situated use of the artifacts, not through reading about multi-agent systems.

Krueger's Responsive Environments and VIDEOPLACE [11], dating to the late 1960s, demonstrated the fundamental loop decades before any of this vocabulary existed: cameras perceive human bodies in a room, and projectors respond in real time. The OAK-D detectors running person detection and gaze tracking are doing what Krueger did, with better models and a coordination layer he never needed because he was working with a single camera and a single response surface.

## Dynamicland

Dynamicland [12], Bret Victor's building-scale communal computer in Oakland, is the most direct inspiration for this classroom and the closest existing system. In Dynamicland, ceiling-mounted cameras and projectors recognize ordinary physical objects — index cards, books, 3D prints — and illuminate them with computational behavior. Programs are physical objects. The entire system runs on a single tightly integrated runtime called Realtalk, and all development happens within Realtalk itself, not on laptops.

The parallels are obvious: cameras perceive the room, physical objects are computational agents, multiple people collaborate simultaneously in a shared space. Both systems treat the room, not the screen, as the primary computational surface.

The differences are architectural and pedagogical. Dynamicland runs on a monolithic runtime — every object is a Realtalk program, and joining the system means writing Realtalk code. The classroom uses a federated architecture: new agents join by connecting to the event bus over HTTP, in whatever language and framework the student prefers. A p5.js sketch in a browser, a Python script on a Raspberry Pi, and a Discord bot are all first-class agents on the same bus. This federation is not a compromise — it is the design. The classroom is a pedagogical environment where students build the agents themselves, and they need to be able to build in whatever they already know.

Dynamicland is a research laboratory where a small team builds a unified vision. The classroom is a teaching environment where eleven students each build a piece of a system that has to cohere without anyone having designed the whole thing in advance. The event bus and the orchestrator exist precisely because there is no Realtalk — no single runtime that holds everything together.

## Multi-Agent Foundations

Minsky's *Society of Mind* [13] proposed that intelligence emerges from vast numbers of individually simple agents organized into teams and hierarchies. The classroom is a modest instantiation of this: camera agents, student project agents, the orchestrator, and humans — each limited on its own, collectively producing behavior that looks like a room that understands what it is doing.

Wooldridge and Jennings [14] defined the canonical agent properties: autonomy, reactivity, proactivity, and social capability. The classroom distributes all four across a heterogeneous population. Camera agents are reactive and social (they detect and publish). The orchestrator is proactive (it reasons about phase and routes accordingly). Student projects vary — some are purely reactive, some incorporate LLMs and plan. Humans are autonomous in ways the system cannot model. No single agent has all four properties. The system as a whole does.

Brooks' subsumption architecture [15] — intelligent behavior emerging from layered reactive behaviors without world models — describes the detector agents accurately. They sense and respond without maintaining models of the room. The orchestrator adds a deliberative layer on top, creating the hybrid reactive-deliberative architecture that the robotics community argued about in the 1990s, implemented here with web infrastructure instead of robot controllers.

The event bus itself is a blackboard system in the sense of Hayes-Roth [16]: a shared information space where independent knowledge sources communicate indirectly, monitoring the blackboard and contributing when their expertise is relevant. The addition of context-aware routing — where the orchestrator decides which events are salient given the current state — extends the classic blackboard with content-based filtering that early blackboard systems lacked.

## LLM Multi-Agent Frameworks

The current wave of multi-agent systems — AutoGen [17], CrewAI [18], LangGraph [19], applications built on MCP [20] and the A2A protocol [21] — assumes all agents are software processes, typically LLM-backed, communicating through function calls, tool use, or structured message passing. These frameworks have advanced the field rapidly, particularly in demonstrating that complex tasks can be decomposed across specialist agents with a supervisor coordinating.

CrewAI's supervisor-specialist pattern closely resembles the classroom's orchestrator-detector architecture. LangGraph's state-machine approach to agent workflow mirrors the orchestrator's phase logic. MCP and A2A address the same interoperability problem the event bus solves: how do heterogeneous agents discover each other and coordinate?

But all of these operate entirely in the digital domain. Their agents are processes. Their "environment" is a conversation thread or a shared memory buffer. They have no cameras, no physical objects, no humans on the bus. The classroom's contribution is not a better agent framework — it is evidence that the same coordination patterns work when the agent population includes things that sense physical reality, things built by students who had never heard the term "multi-agent system," and people sitting in chairs.

## Context-Aware Computing

Schilit, Adams, and Want [22] defined context-aware computing in 1994 as software that "adapts according to the location of use, the collection of nearby people, hosts, and accessible devices." Dey and Abowd [23] broadened the definition: context is "any information that can be used to characterize the situation of an entity," encompassing identity, activity, time, and location. The orchestrator's context state — temporal, social, task, and historical dimensions — is a direct implementation of Dey's framework, applied to event routing rather than application adaptation.

The routing mechanism itself is a form of content-based publish/subscribe [24], where subscribers receive events based on the event's content and the subscriber's declared capabilities, not just a channel name. Traditional topic-based pub/sub (subscribe to "fatigue" and get every fatigue event) is too coarse for a room where the same event type means different things in different contexts. Content-based filtering, modulated by the orchestrator's context state, is what makes the routing context-aware rather than merely categorical.

## Smart Classrooms with Computer Vision

Recent surveys of AI in smart classrooms [26] document a growing ecosystem of camera-based sensing: engagement detection, attention monitoring, emotion recognition. Kim's work on ambient intelligence in classrooms [27] used ambient sensors to detect student engagement and adapt learning strategies. Canedo and Trifan [28] tracked gaze direction and facial expressions to infer attentiveness.

These systems are technically similar to the classroom's detector agents — the same models, the same signals. The architectural difference is that these systems treat cameras as data sources feeding a centralized controller. In the classroom, cameras are agents on a bus. They publish events. Other agents subscribe. The orchestrator routes. No central controller owns the pipeline end-to-end. This decentralization is not an academic preference — it is a practical requirement. When eleven student projects need to consume camera data in eleven different ways, a centralized pipeline cannot serve them all. The bus can.

## Constructionism

Papert's constructionism [29] — the theory that learning is most effective when people build tangible things in the real world — is the pedagogical backbone. The students did not study multi-agent systems and then implement one. They each built a project (a timer, a grammar coach, a projected forest, a virtual dog) and discovered, over the course of a semester, that their projects were agents on a shared bus, that the bus needed context to route effectively, and that an orchestrator was the missing piece. The multi-agent framing emerged from the practice, not the other way around. The event bus is the medium through which student-constructed artifacts become part of the room's intelligence.

## Positioning

No prior work combines all of the following: a federated event bus (not monolithic like Dynamicland), a mixed agent population of sensors, software, and humans (not all-software like AutoGen), context-aware routing that modulates event salience by situational frame (not flat pub/sub), and student-constructed agents as first-class participants whose design was not known in advance. The closest single ancestor is Gaia, updated for an era of LLM-backed orchestration and heterogeneous agent populations. The classroom is not a framework — it is a semester-long experiment in what happens when you build one of these systems with real cameras, real students, and real coordination problems.
