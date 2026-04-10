# Assignment 2: Lane Detection

## Introduction

The scope of this assignment is to develop a lane detection algorithm that works with real images from **PandaSet** data set, available at:
https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset

The archive with the images to be used is available at the following link (17.26 GB):
https://www.dropbox.com/scl/fi/sch1t7ns9vfpa22setcfw/archive.zip?rlkey=0sglcvm9l9xbzb81zoerkhx0b&dl=0

### Camera Parameters

```python
focalLength    = [1970, 1970]
principalPoint = [970, 483]
imageSize      = [1920, 1080]
height         = 1.6600        # camera.Position[2]
pitch          = 0             # camera.Rotation[1]
```

---

## Assignment

1. Develop a **Python 3** implementation of the **GOLD lane detection algorithm**.

2. The code shall receive as input the directory containing the images to be processed (e.g., `PandaSetSensorData/archive/008/Camera/front_camera/`).

3. **(Mandatory requirement)** The code shall process each image in the path, and produce the following output:
   - The image shall be displayed, and the detected lane delimiter(s) shall be displayed as **green vertical line(s)** over the **bird's eye view** of the input image.
   - In case both lanes are not found, a message shall be written on the bird's eye view, such as: `"No lanes found"`.

4. **(Optional requirement 1)** The code shall be able to discriminate between **continuous** lane delimiters and **dashed** lane delimiters.

5. **(Optional requirement 2)** The visualization of the lane delimiter shall be performed on the **original image** instead of the bird's eye view.

6. **(Optional requirement 3)** The algorithm shall recognize the presence of **obstacles in the lane** and report the approximate distance.

---

## Instructions

- The report is **individual**.
- Upload your work as a **ZIP file** using *Portale della didattica*.
- The ZIP file must contain:
  - A **PDF file** with a report describing the algorithm implemented.
  - The **Python code** you implemented.
- The code will be tested against the PandaSet data set `front_camera`.
  - The Python script shall be named **`run_gold.py`**, and the search path of the directory containing the images to be processed shall be provided as an argument on the command line, as in the following example:
    ```bash
    python3 run_gold 'PandaSetSensorData/archive/044/Camera/front_camera/*.jpg'
    ```

> **Deadline: April 21st, 2026 at 11:30**