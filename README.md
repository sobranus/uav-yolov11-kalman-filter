# uav-yolov11-kalman-filter

# UAV Object Distance and Position Estimation using Kalman Filter and YOLOv11 object detection

This repository contains a complete implementation of a real-time **object tracking and distance estimation system** for UAV (drone) applications. The system combines **YOLOv11** object detection, **Kalman Filter–based smoothing**, and **triangulation from dual perspectives** to achieve robust position estimation from aerial video streams.

---

## 📘 Overview

The project estimates:
- **Object distance** relative to the drone camera, derived from object bounding box size and frame scale calibration.
- **Object 2D position** (top-view coordinates) using **stereo triangulation** from two camera perspectives.
- **Filtered trajectory** using a **Kalman Filter**, which stabilizes noisy detections and provides predictive tracking when detections are lost.

---

## System Architecture

1. **YOLOv11 Inference**  
   Detects target objects (e.g., vehicles, people) from UAV camera feeds.

2. **Kalman Filtering**  
   Applies 1D or 2D Kalman filters to smooth the estimated distance or position.

3. **Triangulation**  
   Combines left–right camera detections to estimate the object’s 2D ground position.

---

## Sample:

![Tracking Example](assets/distance_bb.png)

![Distance estimated](assets/Distance.png)

<p align="center">
  <img src="assets/position_l.png" alt="Left perspective detection" width="45%" />
  &nbsp;
  <img src="assets/position_r.png" alt="Right Perspective detection" width="45%" />
</p>

<p align="center">
  <em>Left: Raw YOLOv11 detections — Right: Kalman-filtered trajectory</em>
</p>

![Position estimated](assets/Position.png)

### Directory Structure

project/
├── cpp/ C++ 1/2d kalman filter
├── python/
│ ├── detections.py # Object detection and bounding box annotation
│ ├── distance.py # 1 Perspective distance estimation kf
│ ├── position.py # 2 Perspective position estimation kf
│ ├── drone_coming.mp4 # sample video
│ ├── detections.jsonl # jsonl file of sample object detection annotation
│ ├── uav_nano_tester.pt # Sample YOLOv11n nano light model
│ └── YOLOv11_uav_best.pt # YOLOv11x model weights
├── .gitattributes # Git LFS tracking config
└── README.md