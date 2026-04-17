# Lane Detection

## Project Contents

The project folder contains the following files:

* `run_gold.py`: The Python script containing the implemented algorithm.
* `report.pdf`: A PDF file with a report describing the algorithm implemented.
* `requirements.txt`: File containing all the package dependencies required.
* `yolov8n.pt`: The YOLOv8 weights file used for object detection.
* `README.md`: This README file.

## Prerequisites

Before running the program, make sure you have Python 3 installed on your system. 

It is recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment (Linux / macOS)
source .venv/bin/activate

# Activate the virtual environment (Windows)
.venv\Scripts\activate
```

## Installation

Once your virtual environment is active, install the required packages:

```bash
pip install -r requirements.txt
```

This will automatically install:
* `opencv-python` (for image processing and display)
* `numpy` (for matrix operations and math)
* `ultralytics` (for YOLOv8 object detection inference)

*(Note: The first time you run the script, YOLOv8 will automatically download the `yolov8n.pt` weights file if it's not already in the folder.)*

## How to Run

To run the program, use the `run_gold.py` script and provide the path to the folder containing the images (or a pattern) as an argument.

**Usage:**
```bash
python3 run_gold.py <path_to_directory_or_pattern>
```

**Example:**
```bash
python3 run_gold.py archive/044/camera/front_camera/
```
*(If you provide a directory path, the script will automatically process all `.jpg` images inside it in alphabetical order).*

## How to Exit

During the playback:
* **Press `q` or `ESC`** on your keyboard to instantly interrupt and close the program.
* Alternatively, you can simply **close the result window** using the X button. 

When the sequence of images finishes, the program will notify you in the terminal. You can press any key to close the window and terminate the execution.