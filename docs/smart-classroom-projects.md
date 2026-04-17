# Smart Classroom Projects

> **Archival snapshot.** This document was created on 03/22/26 from the class FigJam board. Projects have changed since then — go to the retros page for the most up-to-date information.

---

## Darren — Imprint

**Project:** Smart Surfaces

### Overview
A camera mounted in the classroom that reads your handwriting off any surface. Write on your desk, a napkin, a scrap of paper, even your hand — it captures what you wrote and saves it directly to your notes.

When you want to review, your notes can be projected back onto any surface around you. No extra supplies, no separate device to manage.

### Features
- Write on any surface, with specialized whiteboard and table features
- Seamlessly alternate between digital and analog
- Screen focused vs. stylus focused?
- April tags?

### Projection
Project to any surface — who needs individual laptops when you can project a personal screen for every person? Project important times like train schedules, calendars, etc. Move and manipulate projections, screenshots, and texts.

### Other Ideas
- Posture meter *(Feifey copyrighted idea)*
- Human mouse/pointer?

---

## Feifey — Focus Beam

### Overview
Focus Beam is a classroom-mounted conversational machine designed to support lectures that use projected slides. It reads the instructor's hand and arm gestures to detect when they are intentionally pointing at a specific area of the screen. When a pointing gesture is recognized, the machine responds by creating a subtle beam or highlighted area over the part of the slide being discussed.

### Purpose
Clearly follow references during class, especially in larger rooms where gestures can be hard to see from a distance.

### What It Does
It creates a subtle focus light on the part of the projected screen the instructor is pointing at, so students can follow the lecture more easily.

### Location
Lives at the front of the classroom as part of the projection setup, near the projector or above the screen.

### How It Senses
Focus Beam senses gesture — specifically the direction and position of the instructor's hand or arm as they point toward the projected image.

- Translates natural teaching gestures into a visual cue that appears directly on the projection
- Functionality example: Auto detect area of focus, shapes, and sections → read/knowing section where to break & camera reading

### When It's Quiet
Stays quiet when no deliberate pointing gesture is detected. Stops when the gesture ends, so the screen returns to normal.

### Design Philosophy
Focus Beam is a calm machine rather than an intrusive one. It only activates when the instructor makes a deliberate pointing gesture — responding to a clear moment of need instead of constantly demanding attention. Its output is subtle: a soft highlight rather than sound, flashing effects, or spoken feedback. When the gesture stops, the machine becomes quiet again.

### Use Cases
- Text-heavy situations
- Code
- Article/paper sharing

---

## Gordon — Smart Stage

### Overview
An automated classroom stage system that activates when a presenter enters and documents the session.

### How It Works
1. **Detect** — Ceiling cameras detect a person entering the stage area using spatial awareness
2. **Activate** — Lights turn on automatically via smart plug; microphone begins recording
3. **Caption** — Real-time speech-to-text displayed on screen and available via web browser
4. **Timeout** — No person detected for 3–5 min → lights off, recording stops automatically
5. **Summarize** — AI processes the recording into structured lecture notes and key takeaways

### System Architecture
- 3 Ceiling Cameras
- USB Microphone
- Raspberry Pi 5
- Smart Plug + Lamp
- Live Caption Display
- AI Summary

### Core Features

**Auto Lighting**
Camera detects presence in Stage Area. Smart plug turns on the spotlight. No buttons, no switches — purely ambient.

**Auto Recording + AI Notes**
Recording starts when Stage activates. Stops after 3–5 min of no presence. AI generates lecture summary automatically.

**Live Caption**
Real-time speech-to-text for international students. Web-based — open a URL on any device to see live subtitles.

---

## Juju — The Virtual Gus

### Overview
We miss Gus... 🖤
So can we bring him back without actually bringing him back?

### How It Works
It goes both ways:

**At the Studio:**
Shows virtual Gus on the whiteboard area or under the screen during class time. Captures Gus from Kathy's room, uses segmentation to cut him out, and live-plays him in the studio.

**At Kathy's House:**
Senses when Gus is around the pet camera and sends the data to the studio. Poses at the studio can activate changes at Kathy's room.

### When Gus Is Detected
The studio displays the live virtual Gus feed.

### When Gus Is Not Detected
- Give Gus a pet?
- Fetching game
- Snack time
- Trancing — plant related
- Something Gus would enjoy without being scared *(need to discuss with Kathy)*

### References
- [Physical Telepresence — MIT Tangible Media](https://tangible.media.mit.edu/project/physical-telepresence/)
- [Dog Trancing — Pupford](https://pupford.com/blogs/all/dog-trancing?srsltid=AfmBOopQBtHtM6LujjZ0A8ztRtAK0negTLnHQfAc785-wTqZk4zHsitR)

---

## Kathy — NodCheck

**Project:** Non-Verbal Comprehension Feedback

### Overview
A tool that helps teachers gauge student understanding through natural head gestures.

The tool addresses a familiar classroom problem: students often hesitate to admit confusion out loud, especially in front of peers. A head gesture is lower-stakes and more instinctive than raising a hand or speaking up. By capturing these non-verbal cues digitally, the teacher receives a clear signal rather than trying to read subtle body language from across the room.

### How It Works
When a teacher asks "Do you understand?", the system opens a short detection window and uses a webcam to determine whether the student nods (yes) or shakes their head (no).

### In Use
A camera faces the students. The teacher initiates a check-in with a single button press, waits a few seconds, and sees a clear yes or no result on screen. A running tally tracks responses across the session, giving a quick sense of overall comprehension.

### Privacy
Activates when the teacher explicitly triggers it, and reads head motion only for that brief window. It does not continuously monitor students, record video, or identify individuals.

### Design Philosophy
This opt-in, momentary approach reflects a deliberate choice: technology should support the teacher-student interaction, not surveil it.

---

## Kevin — Sleep Detection

### Problem
In many classroom environments, students may become tired or lose focus during long lectures. Instructors often cannot easily detect these moments, especially in larger classrooms. Students who are no longer attentive may miss important information, while instructors may assume the class is still following the lecture.

### Concept
A classroom camera quietly detects when a student might be falling asleep. Instead of sending a private reminder, the system triggers a public projection on the classroom screen.

### What It Senses
A camera placed in the classroom observes students' body posture and facial cues, such as:
- Head tilting downward
- Eyes closed for an extended time
- Lack of movement
- Slumped posture

### Wake Up Mode
When sleep is detected, the classroom projector suddenly displays a visual alert:

> **WAKE UP MODE ACTIVATED**
> *"Hey! Stay with us!"*

A playful animation appears on screen.

### Discussion

**How might a smart classroom detect moments when student attention drops and respond in a way that re-engages the class?**

**Advantages**
- Quickly re-engages the class when attention drops
- The projection is public but not directed at a specific person — it acts as a light reminder for the whole class rather than shaming an individual student

**Potential Concerns**
- Camera systems may misinterpret behaviors (thinking vs. sleeping)
- Continuous monitoring raises privacy questions

---

## Mingyue — Smart Room Finder

### Problem
Students enjoy discussing group assignments in small spaces or working on their own assignments in small rooms. However, to find one of these small rooms, students typically have to carry their laptops and wander through the entire studio.

### Room Atmosphere
- Colorful lighting
- Sofas / lounge seating
- Relaxing music

### Interaction
- System scan: which rooms are occupied? Which are empty?
- Tap screen → reserve room

### Technology
- Overhead camera
- Seat detection
- Occupancy tracking
- Real-time map screen placed at the studio entrance

**Display:**
- Room A – Available
- Room B – Occupied
- Room C – Free in 10 min

### Time-Based Lighting Cues
| Time Remaining | Lighting |
|---|---|
| 5–3 min | Soft lighting |
| 3–1 min | Lights begin to change |
| Last 30 sec | Lights flashing |

---

## Mingyue — Procrastination Room

### Concept
A space where procrastination is allowed.

### Problem
Many students experience a similar situation when working in the studio:
- Getting stuck on a design problem
- Losing creative flow
- Low concentration
- Starting to scroll on their phones

Students usually procrastinate at their desks. The result:
- Procrastination lasts much longer
- It becomes difficult to return to work

**Key Insight:** Procrastination itself is not the problem. *Unbounded* procrastination is.

### What the System Tracks
- Room usage
- Average procrastination time
- Peak usage hours

**Example display:**
- Today's usage: 32 students
- Average stay: 4m 30s

This data can help understand student stress levels and studio usage patterns.

---

## Mingyue — Sleep Detection Chair

### Concept
If a student starts falling asleep during class, the chair automatically activates.

The chair slowly moves the student to the front of the classroom, right in front of the teacher.

---

## Phil — Hand Detection Timer

### Overview
A browser-based countdown timer controlled entirely by hand gestures, built with p5.js and ml5.js Handpose.

Point your webcam at your hand to control the timer — no keyboard or mouse needed. The app detects hand landmarks in real time and maps gestures to timer actions. A gesture must be held for ~1.2 seconds to confirm, preventing accidental triggers.

### What It Senses
Hand gestures via webcam.

### Features
- Real-time hand detection via ml5.js Handpose (TensorFlow.js under the hood)
- Gesture hold mechanic — hold ~1.2s to confirm an action
- Audio feedback: confirmation beeps, 1-minute warning, and a completion jingle
- Color-coded timer: white → amber → red as time runs low
- Live camera preview with hand skeleton overlay (top-right corner)
- Gesture reference guide displayed in the bottom-left corner
- Fully responsive — fills the browser window at any size

### Output
Displays a timer and plays a sound when the timer is almost up, as well as when the timer runs out.

---

## Ramon — E-Wall (E-Course Assistant)

### Project Idea
A singular projector faces a wall or whiteboard where presentations happen, while video sensors read actions from the presenter's stage. Visual gestures symbolize actions to create certain visual cues, making the classroom more dynamic.

It creates not only reactive but interactive moments.

### Hardware
- **Projection Hardware:** An ultra-short-throw projector is ideal to minimize shadows, allowing projection onto walls, floors, or tables
- **Sensors:** Depth cameras (e.g., Kinect) or infrared tracking cameras for high-precision motion detection
- **Software & Processing:** TouchDesigner interprets sensor data to drive visual changes, allowing a table to become a digital interactive surface
- **Calibration:** The projector and camera must be calibrated together to map the digital projection to the physical movement area

### Design Question
*How can I incorporate projection mapping into a classroom that helps students and professors create a more dynamic form of presentation?*

### Problems to Solve
- How can a camera highlight a human presenter as the focus point?
- How will it know the difference between mundane body language and purposeful movements?
- How to calibrate the area of focus?
- How to set it up for light problems (thermal, night vision)?

### Note
> Why not use Wall-E?
> Legally speaking, Disney would end me.

### Research Sources
- Reactive projection into a classroom (General search) — Link
- Reframing classroom sensing — Link

---

## Seren — Mode Changing Room

**Project:** Party / Café / Everywhere

### Problem
The classroom changes all day. The room never does.

People use the same space to work alone, catch up in pairs, hang out in groups, and sit through lectures. Every mode calls for a different atmosphere — but the room always responds the same way: nothing.

### Concept
A classroom that reads the room and sets the vibe before you have to ask. A Raspberry Pi + AI monitors context in real time through cameras, sensors, and HDMI input — and adjusts music, lighting, and screen content automatically.

### How People Use It
- Solo work
- Duo sessions
- Group hangouts
- Presentations

### Scenes

**Scene 01 | Solo → Party Mode**
You're alone in the room. AI notices, says hello, and flips the switch — upbeat music, warm light, disco energy.
*Trigger: occupancy sensor ≥ 1*

**Scene 02 | Duo → Café Vibe**
Two people in the room. AI steps back. Lo-fi music plays softly, warm light dims — feels like a study café.
*Trigger: occupancy sensor ≥ 2*

**Scene 03 | Group → Networking Vibe**
Three or more people. Energetic ambient music, screen shows icebreaker prompts. AI facilitates.
*Trigger: occupancy sensor ≥ 3*

**Scene 04 | Board Activity → Focus Mode**
Someone starts writing on the board. Music fades immediately. Room enters quiet focus mode.
*Trigger: CV — motion near board*

**Scene 05 | Presentation → Full Silence**
HDMI input detected. Everything stops. Lights shift to presentation mode. AI goes completely quiet.
*Trigger: HDMI / screen input active*

---

## Seren — Echo Desk

### Problem
Some students have something to say. They just can't say it out loud.
- Language barriers: can't find the words in time
- Fear of speaking: anxiety in public discussion
- Processing time: thoughts not ready fast enough

These students get excluded from classroom discussion — not because they have nothing to say.

### Concept
**EchoDesk** — a conversational machine that speaks for you.

1. Student types a message → sends to mobile
2. Mobile app: text input interface → relays to Raspberry Pi
3. Raspberry Pi: in the classroom → speaks to the classroom
4. Classroom: message is heard by all

A Raspberry Pi speaker sits in the classroom. Students type — the machine speaks.
*Participation without direct speech.*

### How It Works
Students type a response in a mobile app, and a Raspberry Pi speaker in the classroom speaks the message aloud. The system waits for a natural pause before speaking so it doesn't interrupt the conversation.

### AI Support
An optional AI layer can refine the message for clarity while keeping the student's meaning.
> Example: "maybe accessibility problem" → "I think the issue might be accessibility"

### Shared Question Board
Every message also appears on a shared board for the professor, allowing unanswered questions to remain visible until they are addressed. The conversation doesn't move on without them.

### Design Philosophy
EchoDesk is designed to be calm rather than intrusive — a small speaker, soft voice, short messages, a visual indicator before it speaks. It's not a PA system. It's a quiet participant that only shows up when a student asks it to — and a record of every voice that tried to be heard.

---

## Seren — CalmBall

### Concept
CalmBall is a conversational object that helps students regulate stress during class.

When a student squeezes the ball strongly, it signals stress and triggers a short calming sound or music through the classroom speaker connected to the Raspberry Pi. The system creates a moment of pause, helping both students and the professor slow down the pace of the class when tension rises.

### What It Senses
Pressure from squeezing the ball.

### What It Says
A short calming sound or soft music cue.

### When It's Quiet
- When the ball is not squeezed
- When the classroom is calm

### Reflection
The object is designed to be subtle and supportive, creating small moments of relief without interrupting the class.

---

## Shuyang — Assignment Progress Tracking

### Problem
Students often have lots of assignment deadlines at a time, even when they manually write notes and to-do lists. The opportunity is to design a seamless assignment-tracking system that automatically detects relevant coursework materials, extracts deadlines, and surfaces progress reminders when the user is working on their computer.

### Concept
An Ambient Assignment Tracking System — an embedded smart agent that lives inside a computer environment and automatically tracks assignment progress. Instead of requiring users to actively manage their schedules, the system observes relevant signals in the user's workflow and organizes information automatically.

### System Components
1. Identity Detection
2. Document Analysis
3. Automatic Task Generation
4. Desktop Visualization

### Core Workflow

**User Detection**
Webcam detects the owner's face. System activates when the owner begins working.

**Document Scanning**
The system scans local folders for relevant files. Files containing keywords such as "assignment", "project", or course names are identified.

**Deadline Extraction**
The system detects date patterns in documents. Natural language processing identifies phrases such as:
- "Due: Oct 12"
- "Deadline: Friday"
- "Submission date"

**Task Organization**
Extracted information is converted into a structured to-do list. Tasks are organized by urgency and course.

**Ambient Reminder**
The assignment list appears on the desktop when the user starts working. The system updates progress automatically when files are modified.

### AI Companion
To make the system more engaging and human-centered, the assignment tracker is represented as a small AI companion embedded in the desktop interface. Instead of a traditional interface panel, the agent functions as a visual character that:
- Appears when the user starts working
- Reminds the user of approaching deadlines
- Reacts to progress updates

### Design Goals
- Reduce cognitive load in academic task management
- Create a seamless assignment tracking experience
- Integrate productivity tools directly into users' workflows
- Explore how AI agents can support everyday tasks in subtle ways

---

## Sophie — Forest in the Classroom

### Concept
**Input:** Emotion of people (either voice or facial expression)
**Output:** Plants start to grow on the wall and interact with the environment.

### How It Works
When a person enters and speaks their feeling, the forest responds. A plant grows, entering the ecosystem, coexisting with the creatures within it, then disappearing in its own time.

After the person leaves, the forest continues. The plant remains until its own life cycle ends.

### Thinking Process
**Space is neutral.**
The classroom is a space we take for granted. People enter, learn something, leave. The space itself is transparent, neutral, unimportant.

The installation projects a living forest onto the classroom wall. The forest has its own time, its own seasons, its own ecology.

### Inspiration

**Princess Mononoke (もののけ姫)**
It moves by its own ancient rhythm — as the Forest Spirit walks by day in the form of a deer and dissolves into luminous immensity at night. Eboshi, Moro, Ashitaka — to it, they are nothing more than passing figures in an endless cycle. None of them carry any particular meaning.

**Olafur Eliasson**
> "Personally, I find great inspiration in considering how we humans fit into larger, more-than-human systems that comprise land, water, air, and other species."

> "The longer the visitors stay in the space, however, the more they begin to perceive subtle distinctions."

---

## Yuxuan — Ambient English Feedback Object

### Motivation
We often speak English during class discussions, critiques, and presentations. In these moments, the main focus is usually on communicating ideas clearly — because of this, it is difficult to notice small grammar mistakes while speaking.

After the conversation ends, one sometimes realizes they said something incorrectly. However, the moment for correction has already passed. Small grammar patterns may repeat over time without immediate awareness — because nobody points out the mistakes.

### Design Concept
A small smart object that sits on a student's desk.

The object quietly listens when the student speaks English during class. If a grammar mistake occurs, the object provides a gentle visual suggestion on a small screen. The system offers private, subtle feedback that helps students notice their mistakes in real time.

The interaction is calm and minimal so that it does not disrupt the classroom environment.

### Why a Camera Is Included
It helps the system understand who is actually speaking. In a classroom environment, many people may be talking at the same time — a microphone alone might capture voices from nearby students.

The camera helps detect:
- Mouth movement
- Speaking direction
- Whether the student at the desk is the active speaker

This allows the system to provide feedback only when the correct student is speaking, improving the accuracy of the interaction.

### When It's Quiet
If no mistake is detected, the device remains inactive.
