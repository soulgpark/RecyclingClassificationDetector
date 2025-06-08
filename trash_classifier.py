import argparse
import logging
import threading
import time
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageTk
import mediapipe as mp
from tkinter import Tk, Canvas, Label, NW

logging_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
logger = logging.getLogger(__name__)

WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 700
CANVAS_WIDTH, CANVAS_HEIGHT = 1000, 650
CAM_RESOLUTION = (640, 480)
PREVIEW_RESOLUTION = (400, 300)
PREVIEW_POSITION = (10, 10)
TRASH_SIZE = (150, 150)
BIN_SIZE = 100
DRAG_THRESHOLD = 50
ANIMATION_STEPS = 30
ANIMATION_DELAY = 0.02
DOT_RADIUS = 5
DOT_COLOR_BGR = (0, 0, 255)

RECYCLING_INSTRUCTION: Dict[str, str] = {
    "plastic": "Plastics: Empty contents and dispose.",
    "paper":   "Paper: Remove stickers and tape.",
    "can":     "Can: Rinse with water and crush."
}

ICON_PATHS: Dict[str, str] = {
    "plastic": "플라스틱.png",
    "paper":   "종이.png",
    "can":     "캔.png"
}

mp_hands = mp.solutions.hands
HAND_DETECTOR = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_index_finger_position(frame: cv2.Mat) -> Optional[Tuple[int, int]]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = HAND_DETECTOR.process(rgb)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark[8]
    h, w, _ = frame.shape
    return int(lm.x * w), int(lm.y * h)

def vote_classification(model: YOLO, frame: np.ndarray) -> str:
    res = model.predict(frame, verbose=False)
    probs = res[0].probs.data
    arr = probs.cpu().numpy() if isinstance(probs, torch.Tensor) else probs
    return res[0].names[int(np.argmax(arr))]

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
    crop = frame[y1:y2, x1:x2]
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize(TRASH_SIZE)
    pil_img.save(img_path)
    logger.info(f"Saved cropped image to '{img_path}'")
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(label)
    logger.info(f"Saved label '{label}' to '{txt_path}'")

class TrashSortApp:
    def __init__(
        self,
        root: Tk,
        model: YOLO,
        votes_required: int,
        crop_size: int,
        output_img: Path,
        output_txt: Path,
        camera_index: int = 0
    ):
        self.root = root
        self.root.title("Recycling Classification Detector")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.model = model
        self.votes_required = votes_required
        self.crop_size = crop_size
        self.output_img = output_img
        self.output_txt = output_txt
        self.camera_index = camera_index

        self.votes = []
        self.classified = False
        self.dragging = False
        self.trash_type: Optional[str] = None

        self.canvas = Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
        self.canvas.pack()

        self.bin_positions = {"plastic": (200, 550), "can": (500, 550), "paper": (800, 550)}
        self.bin_icons: Dict[str, ImageTk.PhotoImage] = {}

        self._draw_bins()
        self._create_instruction_label()

        self.trash_id = None
        self.trash_bbox = (0, 0, 0, 0)

        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _draw_bins(self) -> None:
        for btype, (bx, by) in self.bin_positions.items():
            img = Image.open(ICON_PATHS[btype]).resize((BIN_SIZE, BIN_SIZE))
            tkimg = ImageTk.PhotoImage(img)
            self.bin_icons[btype] = tkimg

            x0, y0 = bx - BIN_SIZE/2, by - BIN_SIZE/2
            self.canvas.create_image(x0, y0, anchor=NW, image=tkimg)

            self.canvas.create_text(bx, by + BIN_SIZE/2 + 20, text=btype, font=("Arial", 12, "bold"))

    def _create_instruction_label(self) -> None:
        self.instr_lbl = Label(self.root, text="Classifying...", font=("Arial", 15))
        self.instr_lbl.place(x=10, y=CANVAS_HEIGHT + 10)

    def _reset_trash_image(self, image_path: Path, trash_type: str) -> None:
        if self.trash_id:
            self.canvas.delete(self.trash_id)
        img = Image.open(image_path).resize(TRASH_SIZE)
        tkimg = ImageTk.PhotoImage(img)
        x0 = (CANVAS_WIDTH - TRASH_SIZE[0]) // 2
        y0 = 50
        self.trash_id = self.canvas.create_image(x0, y0, anchor=NW, image=tkimg)
        self.trash_bbox = (x0, y0, x0 + TRASH_SIZE[0], y0 + TRASH_SIZE[1])
        self.trash_image_ref = tkimg
        self.trash_type = trash_type
        inst = RECYCLING_INSTRUCTION.get(trash_type, "Info not found.")
        self.instr_lbl.config(text=f"Type: {trash_type}\n{inst}")

    def _camera_loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error("Unable to open camera")
            return
        preview_id = None

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            pos = get_index_finger_position(frame)
            small = cv2.resize(frame, PREVIEW_RESOLUTION)
            if pos:
                sx = int(pos[0] * PREVIEW_RESOLUTION[0] / CAM_RESOLUTION[0])
                sy = int(pos[1] * PREVIEW_RESOLUTION[1] / CAM_RESOLUTION[1])
                cv2.circle(small, (sx, sy), DOT_RADIUS, DOT_COLOR_BGR, -1)

            if not self.classified:
                label = vote_classification(self.model, frame)
                self.votes.append(label)
                text = f"Votes: {len(self.votes)}/{self.votes_required} (Last: {label})"
                cv2.putText(small, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                if len(self.votes) >= self.votes_required:
                    final = max(set(self.votes), key=self.votes.count)
                    logger.info(f"Final classification: {final}")
                    save_results(frame, final, self.crop_size, self.output_img, self.output_txt)
                    self.classified = True
                    self._reset_trash_image(self.output_img, final)

            else:
                if pos:
                    fx = int(pos[0] * CANVAS_WIDTH / CAM_RESOLUTION[0])
                    fy = int(pos[1] * CANVAS_HEIGHT / CAM_RESOLUTION[1])
                    if not self.dragging and self._over_trash(fx, fy):
                        self.dragging = True
                    if self.dragging:
                        self._drag(fx, fy)
                        dropped = self._try_drop(fx, fy)
                        if dropped:
                            break

            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil = ImageTk.PhotoImage(Image.fromarray(rgb))
            if preview_id is None:
                preview_id = self.canvas.create_image(*PREVIEW_POSITION, anchor=NW, image=pil)
            else:
                self.canvas.itemconfig(preview_id, image=pil)
            self.preview_ref = pil

            time.sleep(ANIMATION_DELAY)

        cap.release()

    def _over_trash(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.trash_bbox
        return x1 < x < x2 and y1 < y < y2

    def _drag(self, x: int, y: int) -> None:
        nx, ny = x - TRASH_SIZE[0]//2, y - TRASH_SIZE[1]//2
        self.canvas.coords(self.trash_id, nx, ny)
        self.trash_bbox = (nx, ny, nx+TRASH_SIZE[0], ny+TRASH_SIZE[1])

    def _try_drop(self, x: int, y: int) -> bool:
        for btype, (bx, by) in self.bin_positions.items():
            if abs(x - bx) < DRAG_THRESHOLD and abs(y - by) < DRAG_THRESHOLD:
                correct = (btype == self.trash_type)
                self._finish_drop(btype, correct)
                return correct
        return False

    def _finish_drop(self, btype: str, correct: bool) -> None:
        if correct:
            self._animate(btype)
            msg = f"Correct in {btype}!"
        else:
            msg = "Wrong bin. Try again."
            x0 = (CANVAS_WIDTH - TRASH_SIZE[0]) // 2
            y0 = 50
            self.canvas.coords(self.trash_id, x0, y0)
            self.trash_bbox = (x0, y0, x0+TRASH_SIZE[0], y0+TRASH_SIZE[1])
        self.instr_lbl.config(text=msg)
        time.sleep(1)
        self.dragging = False

    def _animate(self, btype: str) -> None:
        sx, sy, *_ = self.trash_bbox
        tx, ty = self.bin_positions[btype]
        ex, ey = tx - TRASH_SIZE[0]//2, ty - TRASH_SIZE[1]//2
        dx, dy = (ex - sx)/ANIMATION_STEPS, (ey - sy)/ANIMATION_STEPS
        for _ in range(ANIMATION_STEPS):
            self.canvas.move(self.trash_id, dx, dy)
            self.root.update_idletasks()
            time.sleep(ANIMATION_DELAY)
        self.trash_bbox = (ex, ey, ex+TRASH_SIZE[0], ey+TRASH_SIZE[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="models/best.pt",
                        help="Path to YOLO model")
    parser.add_argument("-v", "--votes", type=int, default=30,
                        help="Frames to vote before finalizing")
    parser.add_argument("-c", "--crop_size", type=int, default=300,
                        help="Crop side length")
    parser.add_argument("-oi", "--output_img", default="current_trash.jpg",
                        help="Output image path")
    parser.add_argument("-ot", "--output_txt", default="trash_type.txt",
                        help="Output text path")
    parser.add_argument("-cam", "--camera", type=int, default=0,
                        help="Camera index")
    args = parser.parse_args()

    model = YOLO(args.model)
    root = Tk()
    TrashSortApp(root, model, args.votes, args.crop_size,
                 Path(args.output_img), Path(args.output_txt),
                 camera_index=args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()