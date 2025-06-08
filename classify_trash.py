import argparse
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time trash classification with YOLOv8 (voting + crop & save)")
    parser.add_argument(
        "--model", "-m",
        default="runs/classify/train12/weights/best.pt",
        help="Path to the classification model file")
    parser.add_argument(
        "--votes", "-v",
        type=int, default=30,
        help="Number of frames to vote before final decision")
    parser.add_argument(
        "--crop-size", "-c",
        type=int, default=300,
        help="Side length (in px) of the square crop to save")
    parser.add_argument(
        "--output-img", "-oi",
        default="current_trash.jpg",
        help="Filename for the cropped image output")
    parser.add_argument(
        "--output-txt", "-ot",
        default="trash_type.txt",
        help="Filename for the classification result text output")
    return parser.parse_args()

def load_model(model_path: str) -> YOLO:
    logging.info(f"Loading model from '{model_path}'...")
    return YOLO(model_path)

def vote_classification(model: YOLO, frame: np.ndarray) -> str:
    results = model.predict(frame, verbose=False)
    probs_obj = results[0].probs
    raw = probs_obj.data
    probs = raw.cpu().numpy() if isinstance(raw, torch.Tensor) else raw
    idx = int(np.argmax(probs))
    return results[0].names[idx]

def save_results(
    frame: np.ndarray,
    label: str,
    crop_size: int,
    img_path: Path,
    txt_path: Path
):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    x1, y1 = max(cx - half, 0), max(cy - half, 0)
    x2, y2 = min(cx + half, w), min(cy + half, h)
    cropped = frame[y1:y2, x1:x2]

    pil_img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize((150, 150))
    pil_img.save(img_path)
    logging.info(f"Cropped image saved to '{img_path}'.")

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(label)
    logging.info(f"Classification label saved to '{txt_path}'.")

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    args = parse_args()
    model = load_model(args.model)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Unable to open camera. Check your connection.")
        return

    window_name = "Trash Classifier"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    votes = []
    logging.info("Starting trash classification. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            label = vote_classification(model, frame)
            votes.append(label)

            status_text = f"Votes: {len(votes)}/{args.votes}  (Last: {label})"
            cv2.putText(
                frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2
            )
            cv2.imshow(window_name, frame)

            if len(votes) >= args.votes:
                final_label = max(set(votes), key=votes.count)
                logging.info(f"Final classification: {final_label}")
                save_results(
                    frame, final_label,
                    args.crop_size,
                    Path(args.output_img),
                    Path(args.output_txt)
                )
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("User requested exit.")
                break

            time.sleep(0.03)

    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()