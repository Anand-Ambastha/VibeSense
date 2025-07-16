import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import shutil

def extract_frames(video_path, frame_rate=1):
    video_name = Path(video_path).stem
    output_dir = Path("frames") / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate)  

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_name = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(frame_name), frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")
    return output_dir

def run_yolo_on_frames(frame_dir):
    model = YOLO(r"model\best (4).pt")
    output_dir = Path("crops") / frame_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list(frame_dir.glob("*.jpg"))
    for frame_path in frame_paths:
        results = model(frame_path, save_crop=True, save=False, conf=0.3)

        for result in results:
            if hasattr(result, "save_dir") and result.save_dir:
                cropped_dir = Path(result.save_dir) / "crops"
                if cropped_dir.exists():
                    for class_folder in cropped_dir.iterdir():
                        if class_folder.is_dir():
                            class_name = class_folder.name.lower()

                            if class_name in ["neckline", "sleeve"]:
                                continue
                            for i, img in enumerate(class_folder.glob("*.jpg")):
                                unique_name = f"{frame_path.stem}_{class_name}_{i+1}.jpg"
                                new_path = output_dir / unique_name
                                img.rename(new_path)

        shutil.rmtree(result.save_dir, ignore_errors=True)

    print(f"Saved cropped images to {output_dir}")
    return output_dir

def process_video(video_path):
    frame_dir = extract_frames(video_path)
    cropped_output_dir = run_yolo_on_frames(frame_dir)
    return cropped_output_dir

if __name__ == "__main__":
    video_path = r"videos\2025-05-27_13-46-16_UTC.mp4"
    final_output = process_video(video_path)
    print(f"\n Pipeline complete Cropped images at: {final_output}")