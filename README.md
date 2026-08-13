# Intrinsic AI for Industry Challenge 2026

### Team Cybertrons08 · UR5e · Computer Vision · Visual Servoing

![ROS 2](https://img.shields.io/badge/ROS_2-Kilted-blue?style=flat&logo=ros)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Perception-5C3EE8?style=flat&logo=opencv&logoColor=white)
![YOLO-OBB](https://img.shields.io/badge/YOLO-OBB_Detection-00FFFF?style=flat)
![UR5e](https://img.shields.io/badge/Robot-UR5e-0052CC?style=flat)
![Gazebo](https://img.shields.io/badge/Environment-Gazebo-FF4500?style=flat)

**Global Rank: 21 · Phase 1**  
**Global Rank: 26 · Qualification Phase · Score: 203/300**

---

## 1. Problem Statement

The AI for Industry Challenge 2026 by Intrinsic required an autonomous **UR5e robotic system** to identify a specified NIC port and insert a cable into the correct port.

The solution had to operate across randomized task configurations and perform the complete pipeline from **visual perception to precise cable insertion**.

---

## 2. Overview

We decomposed the complete task into a modular **perception → geometric reasoning → control** pipeline:

```text
Camera
  │
  ▼
YOLO-OBB Perception
  │
  ├── Cable Detection
  └── NIC / Port Detection
          │
          ▼
   Rail Identification
          │
          ▼
   Target Port Selection
          │
          ▼
 Occlusion-Aware Tracking
          │
          ▼
    Pose / Yaw Alignment
          │
          ▼
  Real-Time Visual Servoing
          │
          ▼
 Coordinated Cable Insertion
```



---

## 3. Demo Video


### Autonomous Cable Insertion Pipeline

[output.webm](https://github.com/user-attachments/assets/36de24fc-e80d-4064-8f24-c4612261eed0)

> **Pipeline:** `Detection → Rail Identification → Target Port → Pose Alignment → Visual Servoing → Insertion`


### Core Techniques

- **YOLO-OBB** for NIC/port and cable detection
- **Spatial & geometric reasoning** for rail and target identification
- **Temporal tracking + virtual target estimation** for port occlusion
- **Image-based pose correction** for yaw/rail alignment
- **Closed-loop visual servoing** for cable positioning
- **Coordinated pitch + downward motion** for final insertion

**Robot:** UR5e  
**Environment:** Gazebo (Qualification Phase) · Intrinsic Flowstate (Phase 1)  


---

## 4. Technical Approach

### 4.1 Target NIC Card & Rail Identification

#### Perception

We used **YOLO with Oriented Bounding Boxes (OBB)** to detect the NIC Ports and obtain their geometric information.

OBB detection provides:
- Port center
- Four corner points
- Port orientation
- Edge geometry

This geometric information is used by downstream target-selection and control modules.

#### Rail Identification

Detected ports are grouped according to their spatial relationship to identify individual rails.

The identified rail positions are then continuously tracked in image space as the robot moves.

```text
Port detections
      │
      ▼
Spatial grouping
      │
      ▼
Rail identification
      │
      ▼
Continuous rail tracking
      │
      ▼
Target rail
```

The target rail is tracked dynamically rather than using a fixed image coordinate.

---

### 4.2 Target Port Detection & Occlusion Handling

After identifying the target rail, the detected ports on that rail are used to determine the required target port.

#### Target / Reference Port Relationship

When both ports are visible, we estimate their relative position using the reference-port dimensions:

$$r_x = \frac{x_t - x_r}{h_r}$$

$$r_y = \frac{y_t - y_r}{h_r}$$

where:
- $(x_t, y_t)$ — target port position
- $(x_r, y_r)$ — reference port position
- $h_r$ — reference port image height

The normalized relationship is averaged over multiple observations and locked.

#### Occlusion-Aware Virtual Target

When the target port becomes occluded but the reference port remains visible, its position is estimated as:

$$\hat{x}_t = x_r + r_x h_r$$

$$\hat{y}_t = y_r + r_y h_r$$

This creates a **virtual target** that allows the visual-servoing controller to continue tracking the target during temporary occlusion.

#### Temporal Consistency

Previous target positions are used to reject large unexpected detection jumps and maintain target identity during tracking.

---

### 4.3 Pose / Yaw Alignment

Before performing the insertion, the cable and target must be properly aligned.

#### Port Orientation

The orientation of the detected port is obtained from the OBB corner geometry.

The robot performs proportional wrist-yaw correction until the required orientation is reached.

#### Target Rail Alignment

For the target rail, the angle of the line joining its two ports is used to estimate the rail orientation.

```text
Port 1 ●────────● Port 2
             θ
```

The robot applies proportional angular correction until the target rail is aligned with the required insertion orientation.

---

### 4.4 Real-Time Visual Servoing

After target localization and pose alignment, the robot uses **closed-loop image-based visual servoing**.

The controller continuously tracks:
- Cable position
- Target port / virtual target position

and computes the image-space error:

$$e_x = x_t - x_c$$

$$e_y = y_t - y_c$$

The error is converted into bounded Cartesian velocity commands:

$$v_x = K_p e_x$$

$$v_y = K_p e_y$$

```text
Target
  ●
  │
  │  ex , ey
  │
  ●
Cable
```

The target position is continuously updated from either direct port detection or the virtual-target estimate during occlusion.

---

### 4.5 Cable Insertion

Once the cable is aligned with the target port:

1. Maintain the target position through real-time visual tracking.
2. Perform the final approach toward the port.
3. Apply coordinated **pitch correction + downward Cartesian motion**.
4. Use robot/contact state during the final insertion stage.
5. Complete the cable insertion.

This produces a closed-loop insertion process rather than relying on a fixed pre-programmed trajectory.

---

## 5. Flowstate — Phase 1


Built an **end-to-end cable manipulation pipeline** in Intrinsic Flowstate:

**Cable Detection → Pose Estimation → Grasping → Pickup → Manipulation → NIC Target Identification → Cable Insertion**

- Used **Flowstate Skills & Services** to compose and execute the manipulation workflow.
- Integrated **camera observations, robot state, task state, and skill execution**.
- Used **Intrinsic Vision Model** for vision-based scene/object understanding.
- Integrated our custom **NIC-port detection, pose alignment, visual servoing, and insertion policy** into the Flowstate workflow.

### Result: **21st Place**


---

## 6. Results

| Metric | Result |
|---|---:|
| Competition | Intrinsic AI for Industry Challenge 2026 |
| Team | Cybertrons08 |
| Robot | UR5e |
| Phase 1 Rank | **21st Place** |
| Qualification Phase Rank | **26th Place** |

---

## 7. Technology Stack

### Robotics
`ROS 2` · `UR5e` · `Cartesian Control` · `Joint Control` · `Visual Servoing`

### Computer Vision
`Python` · `OpenCV` · `YOLO` · `YOLO-OBB` · `HSV Segmentation` · `Geometric Vision`

### Estimation & Control
`Temporal Tracking` · `Geometric Reasoning` · `Proportional Control` · `Target Prediction`

### Simulation & Environment
`Intrinsic Flowstate` · `Gazebo`

---

## 8. My Contribution

My primary contribution to **Team Cybertrons08** focused on the vision-driven manipulation pipeline:

- YOLO-based perception
- Rail identification and tracking
- Target-port selection
- Occlusion-aware target prediction
- Geometric pose estimation
- Yaw alignment
- Real-time visual servoing
- Cable insertion control
- Robustness and failure-case handling

This repository is a personal technical showcase of my contribution to the team solution.

---

## 9. Repository Structure

```text
├── AIC-Intrinsic_Demo_Video.mp4  # Complete solution demo video
├── Models/                       # Trained YOLO-OBB & classification models
│   ├── NIC_Card.pt
│   ├── blue.pt
│   ├── obb_model.pt
│   ├── rail_classifier.json
│   ├── rail_classifier.pkl
│   ├── roll_model.pkl
│   └── roll_model1.pkl
├── Policy/                       # Closed-loop vision & control policy
│   └── MyPolicy.py
├── LICENSE                       # License file
└── README.md                     # Documentation & technical writeup
```
