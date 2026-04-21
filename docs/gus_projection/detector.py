from config import TARGET_LABELS, CONFIDENCE_THRESHOLD

def pick_target_detections(detections):
    results = []
    for det in detections:
        label = det.class_name.lower().strip()
        score = det.confidence
        if label in TARGET_LABELS and score >= CONFIDENCE_THRESHOLD:
            results.append(det)
    return results

def has_gus(detections):
    return len(pick_target_detections(detections)) > 0
