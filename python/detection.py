"""
detection.py
Loads an MP4, runs YOLO on each frame, draws bounding boxes,
writes per-frame detections to detections.jsonl, and saves annotated MP4.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

INPUT_VIDEO = "mavic_2_pro_r.mp4"
OUTPUT_VIDEO = "output_mavic_r.mp4"             # output annotated video
DETECTIONS_JSONL = "detections_mavic_r.jsonl"   # output detection data file
MODEL_PATH = "best.pt"
DEVICE = "cpu"                                  # CPU/GPU if supported

# Drawing Styling
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
DIST_BOX_COLOR = (0, 0, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
LINE_TYPE = cv2.LINE_AA
CONF_THRESHOLD = 0.25


def write_jsonl_header(path: str):
    with open(path, "w") as f:
        pass


def append_detection_jsonl(path: str, record: Dict[str, Any]):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def draw_distance_line_and_box(frame, side, bbox, frame_w, frame_h):
    """
    Draw green line and red box with distance value for each side.
    side: one of 'left', 'right', 'top', 'bottom'
    """
    x1, y1, x2, y2 = bbox

    if side == "left":
        distance = int(x1)
        p1, p2 = (0, y1 + (y2 - y1) // 2), (x1, y1 + (y2 - y1) // 2)
        text_org = (x1 // 4, p1[1] - 10)

    elif side == "right":
        distance = int(frame_w - x2)
        p1, p2 = (x2, y1 + (y2 - y1) // 2), (frame_w - 1, y1 + (y2 - y1) // 2)
        text_org = (frame_w - distance // 2 - 20, p1[1] - 10)

    elif side == "top":
        distance = int(y1)
        p1, p2 = (x1 + (x2 - x1) // 2, 0), (x1 + (x2 - x1) // 2, y1)
        text_org = (p1[0] - 15, y1 // 2)

    elif side == "bottom":
        distance = int(frame_h - y2)
        p1, p2 = (x1 + (x2 - x1) // 2, y2), (x1 + (x2 - x1) // 2, frame_h - 1)
        text_org = (p1[0] - 15, frame_h - distance // 2)

    else:
        return

    # Draw line
    cv2.line(frame, p1, p2, BOX_COLOR, 1, LINE_TYPE)

    # Draw red rectangle box behind distance text
    label = str(distance)
    (tw, th), base = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    rect_tl = (text_org[0] - 2, text_org[1] - th - 2)
    rect_br = (text_org[0] + tw + 2, text_org[1] + 2)
    cv2.rectangle(frame, rect_tl, rect_br, DIST_BOX_COLOR, -1)
    cv2.putText(frame, label, (text_org[0], text_org[1]), FONT, FONT_SCALE, TEXT_COLOR, 1, LINE_TYPE)

    return distance


def main():
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    model = YOLO(MODEL_PATH)
    
    model.model.to(DEVICE) if DEVICE != "cpu" else None

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("Cannot open output VideoWriter")

    write_jsonl_header(DETECTIONS_JSONL)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=CONF_THRESHOLD, imgsz=(w, h), verbose=False)
        detections = []
        if len(results) and results[0].boxes is not None:
            r = results[0]
            boxes = r.boxes
            if len(boxes) > 0:  # ✅ make sure there are detections
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                names = getattr(model, "names", None)

                # choose only the highest confidence detection
                best_idx = int(np.argmax(confs))
                x1, y1, x2, y2 = xyxy[best_idx]
                conf = float(confs[best_idx])
                cls_id = int(cls_ids[best_idx])
                cls_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)

                # Draw main bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)

                # Draw distance lines and red boxes
                distances = {}
                for side in ["left", "right", "top", "bottom"]:
                    d = draw_distance_line_and_box(frame, side, (int(x1), int(y1), int(x2), int(y2)), w, h)
                    distances[side] = d

                # Save detection info
                detections.append({
                    "frame_index": frame_idx,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "distances_px": distances
                })

        # even if no detection, still log an empty frame
        append_detection_jsonl(DETECTIONS_JSONL, {"frame_index": frame_idx, "detections": detections})

        out.write(frame)
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total}")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done. {frame_idx} frames processed.")
    print(f"Output video: {OUTPUT_VIDEO}")
    print(f"Detections JSONL: {DETECTIONS_JSONL}")


if __name__ == "__main__":
    main()