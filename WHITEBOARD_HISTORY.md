# Whiteboard Text History Tracking

**Feature 1: Text History Logging** - Track all OCR detections over time for analysis and conversational features.

## Overview

The full whiteboard reader (`whiteboard_reader_full.py`) now logs every text detection to a history file in **JSONL** (JSON Lines) format:

```
~/oak-projects/whiteboard_history.jsonl
```

## What Gets Logged

Every time text is detected, a JSON entry is appended with:

- **timestamp**: ISO format timestamp (e.g., `2026-02-11T19:25:08.123456`)
- **text_lines**: Array of recognized text strings (e.g., `["Create Magik", "Art Poster"]`)
- **num_regions**: Number of text regions detected (e.g., `2`)
- **avg_confidence**: Average confidence score across all recognitions (0.0-1.0)

## Example History File

```jsonl
{"timestamp": "2026-02-11T19:25:08.123", "text_lines": ["LSREATEMACIK"], "num_regions": 1, "avg_confidence": 0.847}
{"timestamp": "2026-02-11T19:25:10.456", "text_lines": ["LCREATE MACIK"], "num_regions": 1, "avg_confidence": 0.873}
{"timestamp": "2026-02-11T19:25:38.789", "text_lines": ["N.SCHIRMSICAL CLASSICS", "SCHIRMERS LIARARS"], "num_regions": 3, "avg_confidence": 0.821}
```

Each line is a complete JSON object (one per detection event).

## Why JSONL?

✅ **Append-only** - No file rewrites, just add new lines
✅ **Easy parsing** - Each line is valid JSON
✅ **Time-ordered** - Chronological by default
✅ **Efficient** - Stream processing for large files
✅ **Simple queries** - Use standard tools (grep, tail, jq)

## Querying the History

### Get Last 10 Detections
```bash
tail -10 ~/oak-projects/whiteboard_history.jsonl
```

### Count Total Detections
```bash
wc -l ~/oak-projects/whiteboard_history.jsonl
```

### Get All Text from Last Hour
```bash
# Using jq (JSON query tool)
tail -100 ~/oak-projects/whiteboard_history.jsonl | jq -r '.text_lines[]'
```

### Find High-Confidence Detections
```bash
cat ~/oak-projects/whiteboard_history.jsonl | jq 'select(.avg_confidence > 0.85)'
```

### Get Detections in Time Range
```bash
# Get detections after 19:25:00
cat ~/oak-projects/whiteboard_history.jsonl | jq 'select(.timestamp > "2026-02-11T19:25:00")'
```

## Python Analysis Example

```python
import json
from pathlib import Path
from datetime import datetime, timedelta

history_file = Path.home() / "oak-projects" / "whiteboard_history.jsonl"

# Read all history entries
entries = []
with open(history_file, 'r') as f:
    for line in f:
        entries.append(json.loads(line))

# Get detections from last 5 minutes
now = datetime.now()
recent = [e for e in entries
          if datetime.fromisoformat(e['timestamp']) > now - timedelta(minutes=5)]

# Most common text
from collections import Counter
all_text = [text for e in recent for text in e['text_lines']]
most_common = Counter(all_text).most_common(5)
print("Most common text:", most_common)

# Average confidence over time
avg_conf = sum(e['avg_confidence'] for e in recent) / len(recent)
print(f"Average confidence: {avg_conf:.2%}")
```

## Feature Status

- **Feature 1**: Text history logging (JSONL) ✅ Complete
- **Feature 2**: Change detection (compare text over time) ✅ Complete
- **Feature 3**: Conversational messages ("I see new text!") ✅ Complete
- **Feature 4**: Confidence aggregation (combine similar readings) ✅ Complete
- **Feature 5**: Smart feedback ("Text cut off on left side") ✅ Complete

## Use Cases

### 1. **Track Whiteboard Evolution**
See how text changes during a class or meeting:
```
19:25:00 - "Agenda: Discuss Project"
19:30:00 - "Agenda: Discuss Project", "Budget: $5000"
19:35:00 - "Agenda: Discuss Project", "Budget: $5000", "Deadline: March 1"
```

### 2. **Detect Camera Movement**
If text completely changes, camera likely moved:
```
19:25:00 - "Create Magik" (poster)
19:26:00 - "Meeting Room Schedule" (different view)
```

### 3. **Build Confidence Over Time**
Same text detected multiple times = higher confidence:
```
19:25:00 - "LSREATEMACIK" (0.85)
19:25:10 - "LCREATE MACIK" (0.87)
19:25:20 - "CREATE MAGIK" (0.92)
→ Best guess: "CREATE MAGIK" (seen 3x, avg 0.88)
```

### 4. **Debug OCR Issues**
See patterns in misrecognition:
```
"SCHIRMSICAL" often misread as "N.SCHIRMSICAL"
→ Suggests extra characters at start
```

## File Management

### Rotate History Files
For long-running deployments, rotate logs periodically:

```bash
# Rotate daily at midnight (cron job)
0 0 * * * mv ~/oak-projects/whiteboard_history.jsonl ~/oak-projects/whiteboard_history_$(date +\%Y\%m\%d).jsonl
```

### Archive Old History
```bash
# Compress history older than 7 days
find ~/oak-projects -name "whiteboard_history_*.jsonl" -mtime +7 -exec gzip {} \;
```

### Clear History
```bash
# Start fresh (backup first!)
mv ~/oak-projects/whiteboard_history.jsonl ~/oak-projects/whiteboard_history_backup.jsonl
```

## Features 3-5 Details

### Feature 3: Conversational Messages

The camera now speaks in natural language instead of robotic status messages:

**Before:** `TEXT DETECTED (3 regions)`
**After:** `I can see new text on the board: Project Due Monday`

Messages vary to feel natural. Examples:
- "Someone just wrote: Budget $5000"
- "Looks like someone edited the board: 'Mondy' changed to 'Monday'"
- "Whoa, I think the camera moved! I'm looking at completely different text now"
- "Looks like the board was erased - it's blank now"

Works in both console output and Discord notifications.

### Feature 4: Confidence Aggregation

OCR readings are noisy - the same text often reads differently each frame:
```
"LSREATEMACIK" (0.85)
"LCREATE MACIK" (0.87)
"CREATE MAGIK"  (0.92)
```

The `ConfidenceAggregator` maintains a rolling buffer of the last 10 readings and clusters similar text using fuzzy matching (60% similarity threshold). It picks the highest-confidence variant as the consensus.

The console status line shows the consensus text and aggregated confidence:
```
Regions: 2 | Best: "CREATE MAGIK" (89%)
```

Change detection also uses consensus text, making state transitions more stable.

### Feature 5: Smart Feedback

Every 30 seconds, the system analyzes detection quality and provides actionable tips:

- **Low confidence:** "Confidence is low (35%) - try adjusting lighting or moving the camera closer"
- **Text cut off:** "Text may be cut off on the left and bottom - try panning the camera"
- **Tiny regions:** "2 text regions are very small - moving closer might help"
- **Good reads:** "Getting clear reads! Confidence: 91%"
- **Unreadable:** "Detected text regions but couldn't read them - the text may be too blurry or at an angle"

## Technical Details

- **Format**: JSONL (JSON Lines) - one JSON object per line
- **Location**: `~/oak-projects/whiteboard_history.jsonl`
- **Append-only**: File grows over time (no overwrites)
- **Timestamp**: ISO 8601 format with microsecond precision
- **Confidence**: Average of all character-level confidence scores from PaddleOCR
- **Logged every frame**: Captures all detections, not just changes

## Example Session

```bash
# Start whiteboard reader
python3 whiteboard_reader_full.py

# In another terminal, watch history in real-time
tail -f ~/oak-projects/whiteboard_history.jsonl | jq '.'

# Output:
{
  "timestamp": "2026-02-11T19:25:08.123456",
  "text_lines": [
    "CREATE MAGIK"
  ],
  "num_regions": 1,
  "avg_confidence": 0.892
}
{
  "timestamp": "2026-02-11T19:25:10.234567",
  "text_lines": [
    "CREATE MAGIK",
    "Art Exhibition"
  ],
  "num_regions": 2,
  "avg_confidence": 0.867
}
```

---

**Status**: ✅ Features 1-5 Complete
