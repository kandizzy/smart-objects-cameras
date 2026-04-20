# Retro — Kathy · JuJu · Seren · Gus Mode  
### IxD ↔ Kathy's Room (Remote Presence System)

![Project screenshot](/docs/whitepaper/artifacts/cutegus.png)

---

## What We Built

We built a **two-location interactive system** that allows Gus — a dog living in Kathy’s apartment — to feel present in the IxD classroom in real time.

A Viam Rover 2 equipped with a camera is placed in Kathy’s room. The live video feed is streamed over the internet and processed on a separate device (a laptop at IxD), where Gus is isolated from the background and projected onto the classroom wall.

Students can:
- See Gus appear dynamically on the wall  
- Trigger interactions (e.g., treat dispenser)  
- Speak to Gus using a voice phrase (“hey buddy!”)

The system creates a lightweight, ambient connection — a shared moment of presence across physical spaces.

---

## System Architecture (What We Actually Built)

Instead of a single-device setup, we implemented a **distributed system across two networks**:

[Kathy’s Apartment]
Viam Rover 2 (Raspberry Pi + Camera)
↓ (Viam Cloud)
[Internet]
↓
[IxD Classroom]
Python Bridge App (Laptop / Mac)
↓
Projection (Wall)

### Key Design Decision

We **decoupled sensing and rendering**:

- **Sensing (camera, robot):** Kathy’s location  
- **Processing + projection:** IxD classroom  

This allowed us to:
- Avoid hardware limitations on the Raspberry Pi  
- Iterate quickly on visual output  
- Keep projection flexible and high-resolution  

---

## Data Flow (Frame → Gus → Projection)

	1.	Camera feed captured (Viam Rover)
	2.	Frame sent via Viam Cloud API
	3.	Python app fetches frame
	4.	Object detection (dog / Gus)
	5.	Bounding box → crop (Gus only)
	6.	Frame resized to projection canvas
	7.	Output rendered via OpenCV → projector

### Implementation Detail

- **Camera input:** `camera.get_images()` (Viam API)  
- **Detection:** Viam Vision Service (object detection)  
- **Segmentation (MVP):** bounding box crop (not full mask)  
- **Rendering:** OpenCV (`cv2.imshow`) → projector display  

---

## What Worked

### 1. Presence Through Projection

Projecting Gus onto the classroom wall created a strong sense of presence.

- It felt less like a screen  
- More like a **window into another space**  
- Spatial scale mattered more than resolution  

---

### 2. Distributed Architecture

Separating:
- **robot (capture)** and  
- **projection (render)**  

…made debugging and iteration much faster.

It also confirmed that:

> Real-time presence does not require co-located hardware.

---

### 3. Interaction Concept

The voice trigger ("hey buddy!") worked conceptually:

- Simple  
- Memorable  
- Low-friction participation  

---

## What Broke

### 1. Raspberry Pi / Network Setup

- Rover was preconfigured for Viam  
- Could not easily switch WiFi environments  
- Required repeated resets between:
  - IxD network  
  - Kathy’s home network  

→ This significantly slowed iteration.

---

### 2. Latency Uncertainty

We never fully identified where delay occurred.

Possible bottlenecks:
- Camera capture  
- Cloud transmission  
- API fetching  
- Detection processing  
- Projection rendering  

→ The system felt near real-time but wasn’t measured.

---

### 3. Detection Limitations

- Default models detect **“dog”**, not **“Gus”**  
- No custom dataset was trained  

→ Identity-level interaction was missing.

---

### 4. Hardware Unknowns

We did not confirm:

- Whether Rover has built-in mic/speaker  
- Whether additional hardware can be attached  
- Whether GPIO control works within this setup  

---

## What I'd Do Differently

### 1. Lock Architecture Early

Before touching hardware:

- Where does processing happen?
  - Pi vs Laptop  
- What data path?
  - Raw stream vs processed stream  

These uncertainties blocked progress throughout the project.

---

### 2. Separate Systems First

Instead of building everything together:

- Build projection system independently  
- Build treat dispenser independently  
- Then integrate  

Trying to do both at once made neither work fully.

---

### 3. Measure Latency Early

Add timestamps at:

- Capture  
- Receive  
- Render  

→ Understand system constraints early.

---

## Open Questions for the Next Team

### System / Architecture

- Should segmentation run on:
  - Raspberry Pi (edge)?  
  - IxD laptop (current)?  

- What’s the fastest path for video transfer?
  - Direct streaming?  
  - API-based polling?  

---

### Computer Vision

- How can we detect **Gus specifically**, not just “dog”?  
  - Custom dataset?  
  - Fine-tuned model?  

---

### Interaction Design

- What stimuli do dogs respond to best?
  - Voice?  
  - Light?  
  - Movement?  

---

### Hardware

- Does Viam Rover 2 support:
  - Mic / speaker?  
  - External modules?  

- Can we attach and control:
  - A treat dispenser?  

---

## Key Insight

> Presence is not about perfect realism —  
> it's about *responsive connection across space.*

Even a cropped, imperfect projection of Gus created:

- Emotional engagement  
- Shared attention  
- A sense of “being together”  

---

## Next Steps (If Continued)

1. Train a Gus-specific detection model  
2. Replace bbox crop with silhouette segmentation  
3. Add latency measurement system  
4. Prototype standalone treat dispenser  
5. Integrate voice + hardware interaction loop  

---

## One-line Summary

A distributed system that streams, isolates, and projects a remote dog into a shared physical space — creating a real-time sense of presence across distance.