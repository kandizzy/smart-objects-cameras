# Whiteboard OCR Reader for OAK-D

Optical Character Recognition (OCR) system for reading text from whiteboards using the OAK-D camera and PaddlePaddle OCR models.

## Two Versions Available

| Feature | `whiteboard_reader.py` | `whiteboard_reader_full.py` |
|---------|----------------------|---------------------------|
| **Text Detection** | ✅ Yes (finds text regions) | ✅ Yes (finds text regions) |
| **Text Recognition** | ❌ No (detection only) | ✅ **Yes (extracts text content)** |
| **Output** | "3 text regions detected" | "Project Due Monday" |
| **Performance** | Faster (single-stage) | Slower (two-stage) |
| **Use Case** | "Is there text?" | "What does it say?" |
| **Complexity** | Simple | Complex |

**Choose based on your needs:**
- **Detection only** (`whiteboard_reader.py`) - Fast, tells you IF text is present
- **Full recognition** (`whiteboard_reader_full.py`) - Slower, tells you WHAT the text says

## Overview

The whiteboard reader uses a **two-stage OCR pipeline**:

1. **Text Detection** (Stage 1): Finds text regions in the camera frame using PaddlePaddle text detection
2. **Text Recognition** (Stage 2): Reads the detected text using PaddlePaddle text recognition

This implementation is adapted from Luxonis oak-examples and follows the established patterns from `person_detector.py` and `fatigue_detector.py`.

## Handwriting Support

**Good news:** PaddleOCR models support both printed AND handwritten text!

- ✅ **Printed text**: Signs, posters, whiteboards with markers
- ✅ **Handwritten text**: Whiteboard writing, notes, markers
- ✅ **Vertical text**: Chinese, Japanese, rotated text
- ✅ **Rotated/curved text**: Non-straight text
- ⚠️ **Cursive handwriting**: Supported but less accurate

The PaddlePaddle models (especially newer PP-OCRv5) are trained on diverse datasets including handwriting, so they should work well with typical whiteboard marker writing.

## Features

- ✅ Real-time text detection from whiteboard
- ✅ Discord bot integration for interactive queries
- ✅ Status file for Discord commands (`!whiteboard`, `!screenshot`)
- ✅ Temporal smoothing to prevent false detections
- ✅ Live video display with text bounding boxes
- ✅ Multi-user coordination (announces who's using the camera)
- ✅ Hardware-optimized models (runs on OAK-D VPU)

## Setup

### Prerequisites

The whiteboard reader requires the same dependencies as other detector scripts:

- DepthAI 3.x (`depthai`)
- DepthAI Nodes (`depthai-nodes`)
- OpenCV (`opencv-python`)
- Python 3.10+

These are already installed in the shared venv at `/opt/oak-shared/venv/`.

### File Structure

```
~/oak-projects/
├── whiteboard_reader.py           # Detection only (simple, fast)
├── whiteboard_reader_full.py      # Full text recognition (complex, slower)
├── depthai_models/                # Model YAML files (copy to Pi)
│   ├── paddle_text_detection.RVC2.yaml
│   ├── paddle_text_detection.RVC4.yaml
│   ├── paddle_text_recognition.RVC2.yaml
│   └── paddle_text_recognition.RVC4.yaml
├── whiteboard_status.json         # Status file (auto-generated)
└── latest_whiteboard_frame.jpg    # Screenshot (auto-generated)
```

### Copying to Raspberry Pi

```bash
# From your laptop (in the smart-objects-cameras directory)

# Copy detection-only version
scp whiteboard_reader.py orbit:~/oak-projects/

# Copy full text recognition version
scp whiteboard_reader_full.py orbit:~/oak-projects/

# Copy model files (required for both versions)
scp -r depthai_models orbit:~/oak-projects/
```

Or use VS Code Remote SSH to edit directly on the Pi.

## Usage

### Basic Usage - Detection Only (whiteboard_reader.py)

```bash
# Activate shared venv
activate-oak

# Basic OCR detection (console output)
python3 whiteboard_reader.py

# With live video display (requires VNC or X11)
python3 whiteboard_reader.py --display

# With file logging
python3 whiteboard_reader.py --log

# With Discord notifications
python3 whiteboard_reader.py --discord

# Quiet mode (only notify when text appears, not when cleared)
python3 whiteboard_reader.py --discord --discord-quiet
```

**Output:**
```
Text regions: 3 | Detected: YES
```

### Full Text Recognition (whiteboard_reader_full.py)

```bash
# Activate shared venv
activate-oak

# Full OCR with text extraction
python3 whiteboard_reader_full.py

# With live video display showing recognized text
python3 whiteboard_reader_full.py --display

# With Discord notifications (includes actual text content)
python3 whiteboard_reader_full.py --discord

# Adjust confidence threshold (filter low-confidence results)
python3 whiteboard_reader_full.py --confidence 0.4
```

**Output:**
```
Regions: 3 | Text: "Project Due Monday"

TEXT DETECTED (2 lines):
  1. Project Due Monday
  2. Submit by 5pm
```

### Command-Line Options

**Both versions:**

| Option | Description |
|--------|-------------|
| `--log` | Log detected text to file |
| `--discord` | Enable Discord notifications for text changes |
| `--discord-quiet` | Only send notifications when text appears (not when cleared) |
| `--display` | Show live window with text bounding boxes |
| `--fps-limit N` | FPS limit (default: 5 for RVC2, 30 for RVC4) |
| `--device ID` | Optional device ID or IP |

**Full version only:**

| Option | Description |
|--------|-------------|
| `--confidence N` | Minimum confidence threshold for recognition (default: 0.25) |

### Examples

**Monitor whiteboard with Discord announcements:**
```bash
python3 whiteboard_reader.py --discord
```

**Debug mode with visual display and logging:**
```bash
python3 whiteboard_reader.py --display --log
```

**Production mode (quiet, Discord only on detection):**
```bash
python3 whiteboard_reader.py --discord --discord-quiet
```

## Discord Integration

### Automatic Announcements

The whiteboard reader automatically announces camera usage to Discord:

**On Startup:**
```
📋 alice is now running whiteboard_reader.py on orbit
```

**When Text Detected:**
```
📝 Text detected on whiteboard (3 regions)
```

**When Whiteboard Cleared:**
```
🗑️ Whiteboard cleared
```

**On Shutdown:**
```
📴 alice stopped whiteboard_reader.py on orbit - camera is free
```

### Discord Bot Commands

The script writes status to `whiteboard_status.json` which the Discord bot can read:

**Status File Format:**
```json
{
  "text_detected": true,
  "text_content": [],
  "num_text_regions": 3,
  "timestamp": "2026-02-11T14:30:00",
  "running": true,
  "username": "alice",
  "hostname": "orbit"
}
```

**Suggested Discord Bot Commands:**
- `!whiteboard` - Show current whiteboard status
- `!read-text` - Get detected text (when implemented)
- `!screenshot` - Get latest whiteboard image with text boxes

## How It Works

### Two-Stage Pipeline

```
Camera (1152x640)
    ↓
Resize to 576x320
    ↓
[Text Detection Model]  ← PaddlePaddle detection (finds text regions)
    ↓
Text regions with bounding boxes
    ↓
[Text Recognition Model] ← PaddlePaddle recognition (reads text)
    ↓
Recognized text + confidence scores
```

### Temporal Smoothing

The script uses a **5-frame rolling window** to smooth detection results:

- Text must be detected in at least 3 out of 5 recent frames
- Prevents false detections from camera shake or reflections
- Makes detection more stable for whiteboard scenarios

### Debouncing

State changes are debounced for **2 seconds**:

- Prevents notification spam when text is being written
- Only triggers alerts when text persistently appears/disappears
- Balances responsiveness with stability

## Testing

### Test 1: Basic Detection

```bash
# Run with display to see what the camera sees
python3 whiteboard_reader.py --display

# Point camera at whiteboard with text
# You should see green boxes around detected text regions
# Console shows: "Text regions: 3 | Detected: YES"
```

### Test 2: Discord Integration

```bash
# Ensure .env file has DISCORD_WEBHOOK_URL
python3 whiteboard_reader.py --discord

# Check Discord for startup message
# Write on whiteboard - should see detection notification
# Erase whiteboard - should see cleared notification
```

### Test 3: Dynamic Updates

```bash
# Start with empty whiteboard
python3 whiteboard_reader.py --discord --log

# Slowly write text on whiteboard
# Observe detection after ~2 seconds of text being visible

# Erase whiteboard
# Observe cleared notification after ~2 seconds
```

### Test 4: Status File

```bash
# Start the reader
python3 whiteboard_reader.py

# In another terminal, watch status file updates
watch -n 1 cat ~/oak-projects/whiteboard_status.json

# Verify status updates as you write/erase text
```

## Troubleshooting

### Model Files Not Found

**Error:**
```
ERROR: Detection model not found at .../paddle_text_detection.RVC2.yaml
```

**Solution:**
```bash
# Copy model files from this repo to Pi
scp -r depthai_models orbit:~/oak-projects/
```

### No Text Detected

**Possible Causes:**
1. **Camera too far from whiteboard** - Move closer (within 1-2 meters)
2. **Poor lighting** - Ensure whiteboard is well-lit
3. **Low contrast** - Use dark marker on white board
4. **Glare/reflections** - Adjust camera angle to minimize reflections
5. **Text too small** - Write larger text (at least 1-2 inches tall)

**Debug Steps:**
```bash
# Use display mode to see what camera sees
python3 whiteboard_reader.py --display

# Check if green boxes appear around text
# If no boxes, adjust camera position/lighting
```

### Performance Issues

**Symptom:** Slow frame rate, laggy detection

**Solutions:**
```bash
# Lower FPS limit
python3 whiteboard_reader.py --fps-limit 3

# Check CPU usage
htop

# Check if thermal throttling
vcgencmd get_throttled
```

### Discord Notifications Not Working

**Solution:**
```bash
# Check .env file exists
cat ~/oak-projects/.env

# Should contain:
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Test notifier directly
python3 discord_notifier.py "Test message"
```

## Comparison to Other Detectors

| Feature | Person Detector | Fatigue Detector | Whiteboard Reader |
|---------|----------------|------------------|-------------------|
| **Pipeline** | Single-stage (YOLO) | Two-stage (Face detect + Landmarks) | Two-stage (Text detect + Recognition) |
| **Model Source** | Luxonis Hub | Luxonis Hub | Luxonis Hub |
| **Primary Use** | Detect people | Monitor student alertness | Read whiteboard text |
| **Output** | Count, bounding boxes | Fatigue %, eye/head state | Text regions, text content |
| **Debounce** | 1.5 seconds | 1.5 seconds | 2.0 seconds |
| **Status File** | `camera_status.json` | `fatigue_status.json` | `whiteboard_status.json` |

## Implemented Features (whiteboard_reader_full.py)

The full reader includes five integrated features:

| Feature | Status | Description |
|---------|--------|-------------|
| **1. Text History** | ✅ | JSONL logging of every OCR reading for time-series analysis |
| **2. Change Detection** | ✅ | Detects NEW, EDITED, REMOVED, CAMERA_MOVED, MIXED changes |
| **3. Conversational Messages** | ✅ | Natural language announcements instead of robotic status |
| **4. Confidence Aggregation** | ✅ | Rolling buffer clusters similar readings for consensus text |
| **5. Smart Feedback** | ✅ | Actionable tips about camera position, lighting, distance |

See `WHITEBOARD_HISTORY.md` for detailed descriptions of each feature.

## Future Enhancements

**Possible Student Projects:**

1. **Text-to-Speech**
   - Read detected text aloud
   - Useful for accessibility
   - Use `pyttsx3` or cloud TTS

4. **Multi-Board Support**
   - Track multiple whiteboards with multiple cameras
   - Aggregate text from all boards
   - Discord command: `!read-all-boards`

5. **Snapshot History**
   - Save whiteboard state periodically
   - Create timelapse of whiteboard changes
   - Discord command: `!history`

6. **OCR Confidence Filtering**
   - Filter low-confidence recognition results
   - Highlight uncertain text in different color
   - Ask for manual verification via Discord

## Technical Details

### Models Used

**Text Detection Model:**
- Name: `luxonis/paddle-text-detection:320x576`
- Input: 320x576 RGB image
- Output: Bounding boxes for text regions
- Platform: RVC2 (Myriad X) and RVC4

**Text Recognition Model:**
- Name: `luxonis/paddle-text-recognition:320x48`
- Input: 320x48 RGB image (cropped text region)
- Output: Character sequence with probabilities
- Platform: RVC2 (Myriad X) and RVC4

### Pipeline Nodes (DepthAI 3.x)

1. **Camera Node** (`dai.node.Camera`)
   - Captures 1152x640 frames at configurable FPS
   - Outputs BGR888 format

2. **ImageManip Node** (Resize)
   - Resizes to text detection model input (576x320)
   - Non-blocking, no frame reuse

3. **ParsingNeuralNetwork** (Detection)
   - Runs PaddlePaddle text detection model
   - Outputs normalized bounding boxes

4. **Future: GatherData Node** (for full pipeline)
   - Would sync detection + recognition results
   - Currently simplified to detection only

### DepthAI 3.x API Notes

This script follows depthai 3.x conventions:

- ✅ Uses `dai.node.Camera` (not deprecated ColorCamera)
- ✅ Creates output queues directly from node outputs (no XLinkOut)
- ✅ Uses `device.getPlatform().name` (not deprecated getMxId)
- ✅ Loads models from Luxonis Hub via YAML descriptors
- ✅ Uses `ParsingNeuralNetwork` from depthai-nodes

## Resources

- **Luxonis Hub Models**: https://models.luxonis.com
- **DepthAI 3.x Docs**: https://docs.luxonis.com/software-v3/depthai/
- **PaddleOCR**: https://github.com/PaddlePaddle/PaddleOCR
- **Original Example**: `/opt/oak-shared/oak-examples/neural-networks/ocr/general-ocr/`

## Credits

Adapted from Luxonis oak-examples general-ocr by the Smart Objects class at SVA.

Uses PaddlePaddle OCR models from Luxonis Hub.
