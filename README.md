# Cat Dataset Collector

This repository contains a Python script designed to run on a Raspberry Pi to automatically collect and annotate images of cats (and optionally people). The resulting dataset is formatted for YOLO and will be used to fine-tune an edge computer vision model, ultimately powering a target-tracking water turret to deter cats from using the yard as a litter box.

## Features

* **Automated Data Collection:** Uses a YOLO model running on hardware accelerators to detect cats and automatically save the frame and bounding box labels.
* **YOLO-Ready Annotations:** Saves labels in standard YOLO format (`class_id x_center y_center width height`) so the dataset is ready for immediate fine-tuning.
* **Smart Cooldowns:** Implements a 3-second cooldown between saved captures to prevent flooding the dataset with near-identical frames.
* **Push Notifications:** Sends a notification with an image via Pushbullet when a cat is detected (limited to once every 15 minutes).
* **Headless or Display Modes:** Automatically detects if a display is attached (via X11 or Wayland). If available, it shows a live annotated feed with manual capture controls.

## Prerequisites

### Hardware

* Raspberry Pi (configured for edge CV inference)
* Compatible MIPI CSI camera module (I'm using Arducam B0647 utilizing the IMX290 sensor for better low-light performance)
* Hailo AI Accelerator (the script defaults to loading a `yolo26s_hailo_model` from a `../models/` directory)

### Software & Dependencies

Install the required Python packages:

```bash
pip install ultralytics opencv-python httpx python-dotenv

```

*(Note: `picamera2` is typically pre-installed on modern Raspberry Pi OS images.)*

## Configuration

If you want to receive mobile or desktop notifications when a cat is spotted, you need a Pushbullet account.

Create a `.env` file in the root directory of the project and add your API key:

```env
PUSHBULLET_API_KEY=your_api_key_here

```

If this key is omitted, the script will simply skip the notification step and continue saving images.

## Usage

Run the script from the command line:

```bash
python collect.py

```

### Command Line Arguments

By default, the script only saves images and labels for **cats**. If you want to include **people** in your dataset to help the model differentiate or track both classes, use the `-p` or `--person` flag:

```bash
python collect.py -p

```

### Keyboard Controls

If you are running the script in a desktop environment (a display is detected), a window will open showing the live feed. You can use the following keys:

* **`s`** - Manually force save the current frame and its annotations (useful if you notice a cat in the frame that hasn't been detected).
* **`q`** - Quit the application cleanly.

## Output Structure

The script automatically generates a `dataset` directory in the current working folder to store the collected data. The class mapping translates standard COCO classes to your custom fine-tuning classes (Cat = `0`, Person = `1`).

```text
dataset/
├── images/
│   ├── frame_1716500000.jpg
│   └── frame_1716500005.jpg
└── labels/
    ├── frame_1716500000.txt
    └── frame_1716500005.txt

```