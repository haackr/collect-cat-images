import os
import time
import cv2
import argparse
from picamera2 import Picamera2
from ultralytics import YOLO

MAIN_DIR = "dataset"
COOLDOWN = 3.0
CLASS_MAP = {
  0: 1,
  15: 0
}

HAS_DISPLAY = os.environ.get('DISPLAY') is not None or os.environ.get('WAYLAND_DISPLAY') is not None

parser = argparse.ArgumentParser(prog="collect_cat_images")
parser.add_argument('-p', '--person', action='store_true')
args = parser.parse_args()
CAPTURE_PEOPLE = args.person

#create the directories
os.makedirs(f"{MAIN_DIR}/images", exist_ok=True)
os.makedirs(f"{MAIN_DIR}/labels", exist_ok=True)

#load the model
model = YOLO('../models/yolo26s_hailo_model')

#setup the camera
picam2 = Picamera2(camera_num=0)
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print(f"Starting Cat{' and Person' if CAPTURE_PEOPLE else ''} Inference...")

last_save_time = 0

def saveImage(timestamp, frame, boxes):
  img_path = f"{MAIN_DIR}/images/frame_{timestamp}.jpg"
  lbl_path = f"{MAIN_DIR}/labels/frame_{timestamp}.txt"
  cv2.imwrite(img_path, frame)
  with open(lbl_path, "w") as f:
          for box in boxes:
            x_c, y_c, w, h = box.xywhn[0].tolist()

            class_id = CLASS_MAP[int(box.cls[0])]

            f.write(f"{class_id} {x_c:.5f} {y_c:.5f} {w:.5f} {h:.5f}\n")

def shouldSave(boxes):
  for box in boxes:
    if int(box.cls[0]) == 15:
      return True
    elif int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
      return True
  return False

try:
  while True:
    classes = [0,15] if CAPTURE_PEOPLE else [15]
    frame = picam2.capture_array()
    results = model(frame, conf=0.15, classes=classes, verbose=False)

    result = results[0]

    timestamp = int(time.time())

    if HAS_DISPLAY:
      anotated_frame = result.plot()
      cv2.imshow("Cat Detector", anotated_frame)

      key = cv2.waitKey(1)

      if key & 0XFF == ord("q"):
        break

      if key & 0XFF == ord("s"):
        print("Manually saving image")
        saveImage(timestamp, frame, result.boxes)

    
    if len(result.boxes) > 0:
      if timestamp - last_save_time > COOLDOWN and shouldSave(result.boxes):
        saveImage(timestamp, frame, result.boxes)
        
        print(f"Target spotted at {timestamp}")
        last_save_time = timestamp

except KeyboardInterrupt:
  print("\nCapture Stopped")

finally:
  picam2.stop()
  if HAS_DISPLAY:
    cv2.destroyAllWindows()
