import argparse
import math
import os
import time
from pathlib import Path

import cv2
import httpx
from dotenv import load_dotenv
from picamera2 import Picamera2
from ultralytics import YOLO

# --- Constants ---
MAIN_DIR = Path("dataset")
COOLDOWN = 3.0
NOTIFICATION_COOLDOWN = 15 * 60
CLASS_MAP = {0: 1, 15: 0}  # COCO to Custom (Person: 1, Cat: 0)
STATIC_TOLERANCE = 0.05

HAS_DISPLAY = (
    os.environ.get("DISPLAY") is not None
    or os.environ.get("WAYLAND_DISPLAY") is not None
)

saved_centers = []


def setup_directories():
    (MAIN_DIR / "images").mkdir(parents=True, exist_ok=True)
    (MAIN_DIR / "labels").mkdir(parents=True, exist_ok=True)


def save_image(timestamp: int, frame, boxes) -> Path:
    img_path = MAIN_DIR / "images" / f"frame_{timestamp}.jpg"
    lbl_path = MAIN_DIR / "labels" / f"frame_{timestamp}.txt"

    cv2.imwrite(str(img_path), frame)

    with open(lbl_path, "w") as f:
        for box in boxes:
            x_c, y_c, w, h = box.xywhn[0].tolist()
            class_id = CLASS_MAP[int(box.cls[0])]
            f.write(f"{class_id} {x_c:.5f} {y_c:.5f} {w:.5f} {h:.5f}\n")

    return img_path


def should_save(boxes) -> bool:
    current_centers = []
    significant_movement = False
    sufficient_detection = False
    for box in boxes:
        is_static = False
        x_c, y_c, _w, _h = box.xywhn[0].tolist()
        current_centers.append((x_c, y_c))

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Check to see if there there is any movement from the detected object.
        for last_x, last_y in saved_centers:
            distance = math.hypot(x_c - last_x, y_c - last_y)
            if distance < STATIC_TOLERANCE:
                is_static = True
                break
        if not is_static:
            significant_movement = True

        # Save if it's a cat (15) OR if it's a person (0) with >50% confidence
        if cls_id == 15 or (cls_id == 0 and conf > 0.5):
            sufficient_detection = True

        # Only save the image if there is significant movement and it's detected at the confidence thresholds.
        if (significant_movement or len(saved_centers) == 0) and sufficient_detection:
            return True

    return False


def send_notification(img_path: Path, timestamp: int):
    api_key = os.getenv("PUSHBULLET_API_KEY")
    if not api_key:
        return

    headers = {"Access-Token": api_key, "Content-type": "application/json"}

    try:
        upload_response = httpx.post(
            "https://api.pushbullet.com/v2/upload-request",
            headers=headers,
            json={"file_name": f"cat_{timestamp}.jpg", "file_type": "image/jpeg"},
        ).json()

        with open(img_path, "rb") as f:
            httpx.post(upload_response["upload_url"], files={"file": f})

        notification_body = {
            "type": "file",
            "title": "Cat Detected!",
            "file_name": upload_response["file_name"],
            "file_type": upload_response["file_type"],
            "file_url": upload_response["file_url"],
            "body": "We saw a motherflippin cat!",
        }

        httpx.post(
            "https://api.pushbullet.com/v2/pushes",
            headers=headers,
            json=notification_body,
        )
    except httpx.RequestError as e:
        print(f"Failed to send notification due to network error: {e}")


def main():
    parser = argparse.ArgumentParser(prog="collect_cat_images")
    parser.add_argument(
        "-p", "--person", action="store_true", help="Include people in captures"
    )
    args = parser.parse_args()
    capture_people = args.person

    load_dotenv()
    setup_directories()

    model = YOLO("../models/yolo26s_hailo_model")

    picam2 = Picamera2(camera_num=0)
    picam2.preview_configuration.main.size = (1280, 720)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    print(f"Starting Cat{' and Person' if capture_people else ''} Inference...")

    last_save_time = 0

    try:
        while True:
            # 0 is Person, 15 is Cat in standard COCO
            classes = [0, 15] if capture_people else [15]
            frame = picam2.capture_array()
            results = model(frame, conf=0.15, classes=classes, verbose=False)

            result = results[0]
            timestamp = int(time.time())

            if HAS_DISPLAY:
                annotated_frame = result.plot()
                cv2.imshow("Cat Detector", annotated_frame)
                key = cv2.waitKey(1)

                if key & 0xFF == ord("q"):
                    break
                if key & 0xFF == ord("s"):
                    print("Manually saving image")
                    save_image(timestamp, frame, result.boxes)

            if len(result.boxes) > 0:
                cat_detected = any(int(box.cls[0]) == 15 for box in result.boxes)

                if timestamp - last_save_time > COOLDOWN and should_save(result.boxes):
                    img_path = save_image(timestamp, frame, result.boxes)
                    print(
                        f"{'Cat' if cat_detected else 'Person'} spotted at {timestamp}"
                    )

                    if (
                        timestamp - last_save_time > NOTIFICATION_COOLDOWN
                        and cat_detected
                    ):
                        send_notification(img_path, timestamp)

                    last_save_time = timestamp

    except KeyboardInterrupt:
        print("\nCapture Stopped")
    finally:
        picam2.stop()
        if HAS_DISPLAY:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
