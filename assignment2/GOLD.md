# GOLD: A Parallel Real-Time Stereo Vision System for Generic Obstacle and Lane Detection

**Authors:** Massimo Bertozzi, Student Member, IEEE; Alberto Broggi, Associate Member, IEEE  
**Published in:** IEEE Transactions on Image Processing, Vol. 7, No. 1, January 1998  
**DOI/Identifier:** S 1057-7149(98)00313-3  
**Affiliation:** Department of Information Technology, University of Parma, I-43100 Parma, Italy  
**Received:** April 5, 1996 | **Revised:** March 24, 1997  
**Supported by:** Italian National Research Council, Progetto Finalizzato Trasporti 2

---

## Abstract

This paper describes the **Generic Obstacle and Lane Detection system (GOLD)**, a stereo vision-based hardware and software architecture designed for use on moving vehicles to increment road safety.

Based on a **full-custom massively parallel hardware**, GOLD:
- Detects **generic obstacles** (without constraints on symmetry or shape)
- Detects **lane position** in structured environments (with painted lane markings)
- Operates at a rate of **10 Hz**

The system leverages a geometrical transform (supported by dedicated hardware) to **remove the perspective effect** from both left and right stereo images:
- The **left image** is used to detect lane markings via morphological filters
- **Both remapped stereo images** are used for detection of free-space in front of the vehicle

Results are displayed on an on-board monitor and a control panel to provide visual feedback to the driver.

**Test results:** The system was tested on the MOB-LAB experimental land vehicle, driven for **more than 3000 km** along extra-urban roads and freeways at speeds up to **80 km/h**. It demonstrated robustness against shadows, changing illumination, different road textures, and vehicle movement.

---

## I. Introduction

The two main issues addressed in this work are:
1. **Lane detection** — implemented using visual data from standard cameras on a mobile vehicle
2. **Obstacle detection** — implemented using the same visual data

### A. Lane Detection

Road following (closing the control loop to keep a vehicle in a safe road position) has been approached in various ways:
- Most systems are based on **lane detection**: compute relative vehicle position, then drive actuators
- Others (e.g., ALVINN) derive steering commands directly from visual patterns, without preliminary lane detection

**Key problems in lane/road boundary detection:**
1. Presence of **shadows** — altering road texture
2. Presence of **other vehicles** — partially occluding road markings

**Common assumptions used to simplify detection:**
- Analysis of specific regions of interest
- Fixed lane width (parallel markings)
- Precise road geometry (e.g., clothoid)
- **Flat road** (assumption used in this work)

**Techniques reviewed:**
- Characteristics of painted lane markings (optionally color-aided)
- Deformable templates: LOIS, DBS, ARCADE
- Edge-based recognition using morphological paradigms
- Model-based approaches: VaMoRs, SCARF
- Model-based analysis for intersection detection
- Optical flow techniques (velocity domain instead of image domain)

### B. Obstacle Detection

The definition of "obstacle" determines the detection technique:

**Narrow definition (obstacle = vehicle):**
- Search for specific patterns based on shape, symmetry, or bounding box
- Can work on a single still image

**Broad definition (obstacle = anything obstructing the path or protruding from the road):**
- Detection is reduced to finding **free-space** rather than recognizing patterns
- Requires two or more images (higher computational complexity)
- Two main techniques:
  1. **Optical flow analysis**: compute ego-motion, then detect obstacles by comparing expected vs. real velocity field
  2. **Stereo image processing**: detect correspondences between two stereo views

**Advantages of stereo vision over optical flow:**
- Directly detects obstacles (vs. indirect inference from velocity field)
- Works even when both vehicle and obstacles have small or null speeds

**GOLD approach:**
- Reduces obstacle detection to determination of **free-space in front of the vehicle**
- No full 3D world reconstruction required
- Uses **Inverse Perspective Mapping (IPM)** on both images (two warpings instead of one)
- Both lane and obstacle detection share the same underlying image warping approach
- Results in the road domain do not need re-projection, simplifying fusion of both functionalities

---

## II. Inverse Perspective Mapping (IPM)

Low-level image processing is efficiently performed on **SIMD (Single Instruction Multiple Data)** systems using a massively parallel paradigm — but only for generic filters that treat images as pixel collections.

**Problem with perspective in road marking detection:**  
Due to the perspective effect, the width of road markings changes with distance from the camera. SIMD systems perform the **same processing on each pixel**, making size-varying template matching impractical.

**Solution:** Remove the perspective effect via IPM, enabling homogeneous pixel processing.

### A. Removing the Perspective Effect

The remapping procedure **resamples the incoming image**, producing a new 2D pixel array representing a **top-down view** of the road region in front of the vehicle.

#### 1. W→I Mapping (World to Image)

**Required parameters:**
- **Viewpoint:** Camera position `(X₀, Y₀, Z₀)`
- **Viewing Direction:** Defined by angles:
  - `α`: angle between projection of optical axis on XY plane and Y axis
  - `θ`: angle between optical axis and versor `ẑ`
- **Aperture:** Camera angular aperture `β`
- **Resolution:** Camera resolution `(W, H)`

**Two Euclidean spaces defined:**
- `W = ℝ³`: 3D world space (world-coordinate)
- `I = ℝ²`: 2D image space (screen-coordinate)

**W→I mapping (forward):**

```
u = (W/2) + (W / 2·tan(β/2)) · [(X - X₀)·cos(α) - (Y - Y₀)·sin(α)] / [(X - X₀)·sin(α)·sin(θ) + (Y - Y₀)·cos(α)·sin(θ) + (Z - Z₀)·cos(θ)]

v = (H/2) + (H / 2·tan(β·H/W/2)) · [(Z - Z₀)·sin(θ) - (X - X₀)·sin(α)·cos(θ) - (Y - Y₀)·cos(α)·cos(θ)] / [(X - X₀)·sin(α)·sin(θ) + (Y - Y₀)·cos(α)·sin(θ) + (Z - Z₀)·cos(θ)]
```

#### 2. I→W Mapping (Inverse — IPM)

The inverse transform removes the perspective effect and recovers the texture of the Z=0 plane (flat road assumption):

```
X = X₀ - Z₀ · [((u - W/2)·cos(α)/tan(β/2)/W) - sin(α)·sin(θ) - ... ] / [...]
Y = Y₀ - Z₀ · [((u - W/2)·sin(α)/tan(β/2)/W) + cos(α)·sin(θ) - ... ] / [...]
```

**Implementation:** Scan pixels `(u,v)` of the remapped image, assign to each the corresponding value from the source image at `(u', v')`.

**Result:** Road marking width becomes nearly **invariant across the whole image**, enabling uniform SIMD processing.

> Note: The lower portion of the remapped image is undefined due to camera position/orientation constraints.

---

## III. Stereo Inverse Perspective Mapping

### Motivation

A 3D description from a single 2D image is impossible without a priori knowledge. Traditional stereo vision requires:
1. Camera calibration
2. Feature localization in one image
3. Feature identification in the other image (correspondence problem)
4. 3D scene reconstruction via triangulation

**Domain-specific constraint used:** Flat road assumption — greatly reduces complexity.

### Horopter and Flat Road

The **horopter** is the zero-disparity surface of a stereo system — objects matching the horopter appear identical in both views.

- For small camera vergence (typical in automotive), the horopter is approximately **planar**
- The horopter cannot be aligned with the road plane using only camera vergence — **electronic vergence (IPM)** is required
- Under the flat road hypothesis, IPM produces top-down views where **pixels with the same coordinates in both remapped images are homologous points** (same real-world point on the road)

**Obstacle detection principle:**  
A generic obstacle (anything protruding from the road) causes disparities in the remapped images. Computing the **difference** between the two remapped images reveals clusters of non-zero pixels where obstacles are present.

### Geometry of Vertical Edges

IPM maps vertical straight lines (perpendicular to road plane) into straight lines passing through the camera projection `P = (X₀, Y₀)` on the road plane:

```
Defining:  k = tan(α) · (u - W/2) / (W / 2·tan(β/2)) / sin(θ)  +  cos(α)
           l = -1/sin(θ) · ...

=> X = k · Y + l  (a straight line through P)
```

Therefore, the **vertical edges of a generic obstacle** are mapped into two straight lines intersecting at the camera projection point. In the difference image, an ideal obstacle produces **two triangles**.

### A. Camera Calibration

Calibration parameters:
- **Intrinsic** (fixed): angular aperture `β`, resolutions `W` and `H`
- **Extrinsic** (measured and tuned): viewpoint `(X₀, Y₀, Z₀)`, angles `α` and `θ`

Two extrinsic parameters (`X₀`, `Y₀`) are shared between both cameras. Parameters `α` and `θ` are determined per-camera by iterative application of stereo IPM, minimizing disparities in remapped images of a flat road while the vehicle is stationary.

**MOB-LAB camera acquisition parameters:** (See Table I in original paper)

---

## IV. Driving Assistance Functions

Both lane detection and obstacle detection are divided into:
- **Low-level phase**: efficiently expressed as SIMD operations (implemented on PAPRICA hardware)
- **High/medium-level phase**: serial processing (implemented on PAPRICA host computer)

### A. Lane Detection

**Assumption:** In the remapped image, a road marking appears as a **quasi-vertical bright line of constant width** surrounded by a darker region.

Pixel belonging to a road marking: brightness higher than neighbors at a given horizontal distance.

#### 1. Parallel Feature Extraction (Low-Level)

**Filtering step:**  
For each pixel `(u, v)` with brightness `f(u,v)`, compute:

```
g(u,v) = min(f(u,v) - f(u-d, v),  f(u,v) - f(u+d, v))   if both differences > 0
        = 0                                                   otherwise
```

where `d` depends on road markings width, acquisition process, and remapping parameters.

**Enhancement step:**  
Geodesic morphological dilation with binary structuring element (cross-shaped, extending vertically):

```
Structuring element B:
  [0 1 0]
  [1 1 1]
  [0 1 0]
```

Control image `c(u,v)`:
```
c(u,v) = g(u,v)    if g(u,v) > 0
        = 0         otherwise
```

The geodesic dilation computes the maximum value in the neighborhood defined by B, masked by the control image. Iterated multiple times, it propagates the maximum value along vertical corridors — pixels at distance `d` from a road marking have zero control value, forming a barrier.

**Binarization step:**  
Adaptive threshold using the maximum value in a local neighborhood:

```
b(u,v) = 1    if e(u,v) ≥ t · max_{(i,j) ∈ N(u,v)} e(i,j)
        = 0    otherwise
```

where `e` is the enhanced image, `t` is a constant, and `N` is the neighborhood. Typical values: `t = 0.7`, neighborhood `= 7×7`.

#### 2. Feature Identification (Medium-Level)

The binary image is scanned **row by row**. Each pair of non-zero pixels can represent one of three road configurations (left edge + right edge, left edge + center, center + right edge).

Each valid configuration produces a pair `(m, w)` where:
- `m` = coordinate of road medial axis
- `w` = corresponding lane width

A **histogram** of `m` values is built per image row. After low-pass filtering, its peak value `ŵ` is found.

All pairs with `|w - ŵ| ≤ threshold` are accepted. The image is scanned **bottom-to-top** (road more visible at bottom), and the **longest chain of road centers** is built exploiting vertical correlation.

The perspective effect is then reintroduced for display using the dual (W→I) transform.

### B. Obstacle Detection

Based on the stereo IPM difference image. Ideal obstacles produce **pairs of triangles** in the difference image; real obstacles produce quasi-triangular clusters.

**Low-level preprocessing:**
1. Compute difference between two remapped images
2. Apply threshold
3. Apply **morphological opening** to remove small-sized details

#### 1. Polar Histogram

**Key geometric property:**  
The prolongations of a triangle's edges pass through the projections `P_L` and `P_R` of the two cameras onto the road plane.

**Construction:**
- Focus placed at the midpoint between `P_L` and `P_R`
- For every angle `φ`, count the number of above-threshold pixels along the line from focus at angle `φ`
- Normalize by reference image (all pixels set)
- Apply low-pass filter to reduce noise

**Triangle detection:**  
Each triangle edge creates a **peak** in the polar histogram. One obstacle → two triangles → **two adjacent peaks**.

Peak characteristics (amplitude, sharpness, width) depend on:
- Obstacle distance
- Angle of view
- Difference in brightness/texture between obstacle and background

#### 2. Peaks Joining

Two adjacent peaks are joined if they are generated by the same obstacle.

**Joining criterion:**  
Let `A1` = area under the curve between peaks (above valley), `A2` = area of the valley below the peaks' base.

```
R = A1 / A2
```

If `R > threshold`, the two peaks are joined (same obstacle).  
If peaks are far apart or valley is too deep → peaks remain separate.

The angle interval between joined peaks determines the **angle of view** of the obstacle.

#### 3. Estimation of Obstacle Distance

For each peak, compute a **radial histogram** by scanning a sector of the difference image:
- Sector width `Δφ` = width of polar peak at 80% of its maximum amplitude
- Count above-threshold pixels along radial lines
- Normalize the result

The radial histogram is analyzed to detect the **corners of triangles**, which represent contact points between obstacles and the road plane — allowing distance estimation via threshold.

Results are displayed as **black markers** superimposed on a brighter version of the left image, encoding both distance and width of each detected obstacle.

---

## V. The Computing Architecture

### Design Constraints

Response time is critical (directly affects maximum allowed vehicle speed). Future trends in mobile computing favor massively parallel architectures with many slow processing elements (PEs).

**Power consumption** of dynamic systems: `P ∝ C · f · V²`

Where:
- `C` = circuit capacitance
- `f` = clock frequency
- `V` = voltage swing

**Power saving strategies:**
1. Greater VLSI integration → reduce `C`
2. Lower clock frequency → trade speed for power
3. Reduce supply voltage `V` (IC voltage reduced from 5V to 3.3V and below)

**CMOS gate delay:** `τ ∝ V / (V - Vth)²`  
→ Reducing `V` increases delay quasi-linearly (until threshold `Vth`)  
→ But reduces power **quadratically**

**Conclusion:** Operate at lowest possible speed, compensate with parallelism.

**Features required for on-board vehicle integration:**
1. Low production cost
2. Low operative cost
3. Small physical size

### PAPRICA System

The **Parallel Processor for Image Checking and Analysis (PAPRICA)** system is a low-cost special-purpose massively parallel architecture developed in cooperation with the Polytechnic Institute of Turin.

**Hardware specs:**
- **256 Processing Elements (PEs)** in SIMD fashion
- Integrated on a single **VME board (6U)** connected to a SPARC-based host workstation
- PE array: **16×16 square matrix** of 1-bit PEs, each with:
  - Full 8-neighbor connectivity
  - Internal 64-bit memory

**Five major functional parts:**
1. **Program memory**: up to 256,000 instructions
2. **Image memory**: up to 8 MB
3. **Processor array**: 16×16 PEs
4. **Camera interface**: stereo image acquisition + external monitor display at video rate (25 fps / 50 fields/s)
5. **Control unit**: manages the entire system

**Virtualization of the Processor Array:**  
Since number of PEs (256) << number of image pixels, subwindows of the image are processed iteratively: load subwindow → process → store results → repeat.

### Key PAPRICA Features for GOLD

#### 1. Morphological Processing
Instruction set includes **graphical** and **logical** operators:
- Graphical operators derived from mathematical morphology (set-theory based bitmap processing)
- For each pixel: input from one bit-plane of the pixel + same bit-plane of 8 neighbors
- Output stored in destination bit-plane or used as first operand of logical operation

#### 2. Support for Pyramidal Vision
- PAPRICA's processor virtualization mechanism allows implementation of **multi-resolution algorithms** without dedicated hardware interconnections
- Avoids the 3D interconnection structure problem of standard pyramid architectures

#### 3. Camera Interface
- Grabs pairs of **8-bit/pixel grey-tone stereo images** directly into PAPRICA image memory
- Display formats:
  - `512×512` pixels at 25 Hz (full frames)
  - `512×256` pixels at 50 Hz (single fields)

#### 4. Data Remapping Support
- A dedicated serial hardware device (implemented in FPGA) handles **global communications** (pixels moving to arbitrary destinations)
- An **address image** (look-up table) is loaded into image memory; each pixel contains a pointer to source data
- Remapping rate: **3 × 50 ns clock cycles per pixel** → ~3 ms for a 128×128 image
- This hardware is fundamental to the GOLD system's IPM functionality

---

## VI. Performance Analysis

### System Architecture

GOLD comprises two independent computational engines running **pipelined**:
- **PAPRICA**: runs low-level processing
- **Host computer (SPARC)**: runs medium-level processing

Total timing = max(PAPRICA time, host time), not their sum.

### Timing Breakdown (per frame cycle)

| Phase | Executor | Time |
|---|---|---|
| **Data Acquisition & Output** | PAPRICA (camera interface) | ~20 ms (one time slot) |
| **Remapping** | PAPRICA (hardware remapping) | 6 ms (two 128×128 images from 512×256) |
| **Obstacle Detection Preprocessing** | PAPRICA | 25 ms + 3 ms transfer |
| **Lane Detection Preprocessing** | PAPRICA | 34 ms + 3 ms transfer |
| **Obstacle Detection (medium-level)** | Host computer | 20–30 ms (data-dependent) |
| **Lane Detection (medium-level)** | Host computer | ~30 ms (data-dependent) |
| **Warnings display** | Control panel | Negligible |

**Notes:**
- Images acquired in single-field mode: 512×256 at 50 fps (20 ms time slots)
- PAPRICA remains idle at end of each slot waiting for next frame
- Whole processing: **5 time slots = 100 ms**
- **System rate: 10 Hz**
- Control panel display latency: 150 ms

---

## VII. Discussion

### Test Summary

The system was tested on **MOB-LAB** (4 cameras, computers, monitors, control panel), driven for **>3000 km** on extra-urban roads and freeways, under different traffic/illumination conditions, at speeds up to **80 km/h**.

GOLD is also being ported to **ARGO** (Lancia Thema passenger car with automatic steering).

### Lane Detection: Failure Conditions

Lane detection fails when:
1. **Road is not flat** → irregular thresholded remapped image
2. **Road markings not visible** → incomplete thresholded remapped image

**Success rate:** ~95% of situations tested (unofficial tests; does not cover all road conditions).

### Obstacle Detection: Robustness to Camera Parameter Drift

Camera calibration is less critical than in full 3D reconstruction systems, since the goal is only **free-space determination**. Dynamic recalibration is generally not needed; recalibration (loading a new look-up table) can be done during host computer idle time.

Tested with varying camera height (`Z₀`) and inclination (`θ`): obstacle detection remained reliable even with noisy difference images.

### Obstacle Detection: Failure Conditions and Tunable Parameters

#### Failure Scenarios

| Scenario | Description |
|---|---|
| Obstacle too far (>45–50 m) | Small/isolated peaks, difficult to join; reliable range: **5–45 m** |
| Guard-rail adjacent to obstacle | Detected as a single large obstacle |
| Partially visible obstacle | Only one edge detected → single peak, not joinable |
| Noisy polar histogram peaks | False small obstacle detections |
| Obstacle brightness ≈ road brightness | Detection of far obstacles may fail |

#### Tunable Parameters

**1. Obstacle Height**  
Determines peak amplitude in polar histogram. The bandwidth of the low-pass filter on the polar histogram is used as a threshold to discard small peaks (noise or short obstacles). Trade-off: smaller bandwidth → less noise sensitivity, but higher minimum detectable obstacle height.

**2. Obstacle Width**  
Determined by the peak-joining ratio threshold `R = A1/A2`. Smaller threshold → wider detectable objects, but higher risk of incorrectly joining peaks from different obstacles.

**3. Obstacle Distance**  
Farther obstacles produce smaller triangle portions and lower polar histogram peaks. For sufficiently high obstacles (~50 m), the main issue is peak joining rather than detection.

**4. Obstacle Shape**  
Algorithm designed for **quasi-vertical edges**. Non-vertical edges (e.g., pyramidal objects) generate twisted triangles that are harder to detect.

### Camera Setup Considerations

- Different setup could increase remapped image resolution and reduce sky-framing issues
- **Inter-camera spacing**: larger spacing → stronger disparities (better detection), but higher sensitivity to vehicle rolling
- Cameras installed at maximum allowed distance given MOB-LAB physical structure

### Temporal Correlation (Future Extension)

At 100 km/h with MOB-LAB calibration: vertical shift between subsequent remapped frames = only **7 pixels** (100 ms interval). This high temporal correlation enables:
- **Time-averaging** of processing results → reduces incomplete obstacle detection
- **Noise reduction** from vehicle movements (modeled as high-frequency sinusoid)

An extension exploiting temporal correlations and deeper fusion of lane/obstacle detection functionalities is under test on **ARGO**.

---

## Key System Parameters Summary

| Parameter | Value |
|---|---|
| Processing rate | 10 Hz |
| Input image size | 512×256 pixels (single field) |
| Remapped image size | 128×128 pixels |
| Remapping time | ~6 ms (both images) |
| Total cycle time | ~100 ms (5 × 20 ms slots) |
| PAPRICA PEs | 256 (16×16 SIMD array) |
| PE memory | 64 bits internal |
| Image memory | Up to 8 MB |
| Program memory | Up to 256K instructions |
| Test distance | >3000 km |
| Max test speed | 80 km/h |
| Lane detection success rate | ~95% |
| Reliable obstacle range | 5–45 m |

---

## References (Selected Key Works)

1. Bertozzi, Broggi, Castelluccio — Real-time vehicle detection system (J. Syst. Architecture, 1997)
2. Bertozzi, Broggi, Fascioli — Obstacle and lane detection on ARGO (IEEE ITS Conf. '97)
3. Beucher & Bilodeau — Road segmentation by fast watershed transform (IEEE Intelligent Vehicles '94)
4. Broggi — Performance optimization on low-cost cellular array processors (MPCS '94)
5. Broggi — Real-time lane and road detection in critical shadow conditions (IEEE ISCV '95)
7. Broggi & Berté — Vision-based road detection: expectation-driven approach (J. Artif. Intell. Res., 1995)
8-9. Broggi et al. — PAPRICA parallel architecture (J. VLSI Signal Process.)
17. Crisman & Thorpe — SCARF: color vision for road tracking (IEEE Trans. Robot. Automat., 1993)
18. Dickmanns & Mysliwetz — Recursive 3D road and ego-state recognition (IEEE TPAMI, 1992)
26. Graefe & Kuhnert — Vision-based autonomous road vehicles (Vision-Based Vehicle Guidance, 1991)
27. Haralick, Sternberg, Zhuang — Image analysis using mathematical morphology (IEEE TPAMI, 1987)
33. Koller, Malik, Luong, Weber — Integrated stereo-based approach to vehicle guidance (ICCV '95)
36. Mallot et al. — IPM simplifies optical flow and obstacle detection (Biol. Cybern., 1991)
43-44. Pomerleau — ALVINN: Neural network based autonomous navigation
45. Pomerleau — RALPH: Rapidly Adapting Lateral Position Handler (IEEE Intelligent Vehicles '95)
53. Serra — Image Analysis and Mathematical Morphology (Academic, 1982)

---

*End of document. Full reference list available in original IEEE publication.*