from ultralytics import YOLO
import cv2
import os
import pandas as pd
from datetime import datetime
from PIL import Image
import torch
from torchvision import transforms

# ============================
# STEP 1: Load YOLO model
# ============================
MODEL_PATH = 'models/yolo_v12/weights/best.pt'
ROI_X1, ROI_Y1 = 820, 80
ROI_X2, ROI_Y2 = 1750, 540
TARGET_CLASSES = ['tray', 'dish']

model = YOLO(MODEL_PATH)
model.fuse()
CLASS_NAMES = model.names

# ============================
# STEP 2: Load classification models
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
dish_classifier = torch.load('models/dish_classifier.pt', map_location=device)
tray_classifier = torch.load('models/tray_classifier.pt', map_location=device)
dish_classifier.eval()
tray_classifier.eval()

# Define class labels
dish_classes = ['empty', 'kakigori', 'not_empty']
tray_classes = ['empty', 'kakigori', 'not_empty']

# Preprocess transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ============================
# STEP 3: Classification helper
# ============================


def classify_crop(crop_img, classifier, class_names):
    pil_img = Image.fromarray(cv2.cvtColor(
        crop_img, cv2.COLOR_BGR2RGB)).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(input_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# ============================
# STEP 4: Tracking + Classification
# ============================


def run_yolo_tracking(input_video_path, conf=0.25, iou=0.5, max_det=100, classes="", output_dir="yolo_results"):
    try:
        class_filter = list(
            map(int, classes.strip().split())) if classes else None
    except ValueError:
        class_filter = None

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_video_path)
    output_video_path = os.path.join(output_dir, f"tracked_{filename}")

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(
            "Error.")

    log_data = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
        results = model.track(
            roi,
            conf=conf,
            iou=iou,
            max_det=max_det,
            persist=True,
            verbose=False
        )[0]

        for box, cls_id, conf_score, track_id in zip(
            results.boxes.xyxy, results.boxes.cls, results.boxes.conf, results.boxes.id
        ):
            class_id = int(cls_id)
            class_name = CLASS_NAMES[class_id]

            if class_name not in TARGET_CLASSES:
                continue
            if class_filter and class_id not in class_filter:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            abs_x1, abs_y1 = x1 + ROI_X1, y1 + ROI_Y1
            abs_x2, abs_y2 = x2 + ROI_X1, y2 + ROI_Y1

            # === Crop object for classification
            crop = roi[y1:y2, x1:x2]
            if class_name == 'dish':
                sub_label = classify_crop(crop, dish_classifier, dish_classes)
            elif class_name == 'tray':
                sub_label = classify_crop(crop, tray_classifier, tray_classes)
            else:
                sub_label = 'unknown'

            # === Annotate frame
            label = f"{class_name} ({sub_label}) ID:{int(track_id)} {conf_score:.2f}" if track_id is not None else f"{class_name} ({sub_label})"
            cv2.rectangle(frame, (abs_x1, abs_y1),
                          (abs_x2, abs_y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (abs_x1, abs_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            log_data.append({
                "frame": frame_count,
                "track_id": int(track_id) if track_id is not None else -1,
                "class_id": class_id,
                "class_name": class_name,
                "sub_label": sub_label,
                "conf": round(float(conf_score), 2),
                "x1": abs_x1,
                "y1": abs_y1,
                "x2": abs_x2,
                "y2": abs_y2,
                "timestamp": datetime.now().isoformat()
            })

        # Draw ROI box
        cv2.rectangle(frame, (ROI_X1, ROI_Y1),
                      (ROI_X2, ROI_Y2), (255, 0, 0), 1)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 1000:
        df_log = pd.DataFrame(log_data)
        return output_video_path, df_log
    else:
        raise RuntimeError("❌ Video kết quả bị lỗi hoặc rỗng.")

# ============================
# STEP 5: Run on image (unchanged)
# ============================


def run_yolo_on_image(img_array):
    image = img_array.copy()
    roi = image[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    results = model.predict(roi, conf=0.25, verbose=False)[0]

    for box, cls_id, conf_score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        class_id = int(cls_id)
        class_name = CLASS_NAMES[class_id]
        if class_name not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        abs_x1, abs_y1 = x1 + ROI_X1, y1 + ROI_Y1
        abs_x2, abs_y2 = x2 + ROI_X1, y2 + ROI_Y1

        label = f"{class_name} {conf_score:.2f}"
        cv2.rectangle(image, (abs_x1, abs_y1),
                      (abs_x2, abs_y2), (0, 255, 0), 2)
        cv2.putText(image, label, (abs_x1, abs_y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.rectangle(image, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (255, 0, 0), 1)
    return image
