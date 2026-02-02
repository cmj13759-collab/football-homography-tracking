# Football Player Tracking And Homography Project

A computer vision pipeline that maps detected football players from video into standardized field coordinates using homography. This enables spatial analysis such as player positioning, spacing, and movement relative to the field.

## Overview
This project takes in Coaches All-22 sideline film and detects field landmarks and computes a planar homography. Then projects player detections into a football field coordinate system.

## Features
- Processes football video frames
- Detects field landmarks (yard numbers, yardlines, sidelines, hash marks)
- Computes a homography between image space and field space
- Projects detected player positions onto a standardized field model
- Outputs annotated visualizations and tracking data


## Tech Stack
- **Language:** Python
- **ML:** PyTorch
- **Computer Vision:** OpenCV, NumPy
- **Utilities:** tqdm, python-dotenv
- **Annotation/Visualization:** supervision
- **Optional (external):** SAM-2 (Segment Anything 2) via `segment-anything-2-real-time`

## Status
Work in progress

## Roadmap
- Add player number identification
- Improve homography stability

