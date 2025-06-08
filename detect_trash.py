import argparse
import logging
import threading
import time
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import mediapipe as mp
from PIL import Image, ImageTk
from tkinter import Tk, Canvas, Label, NW

tlogging_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=tlogging_format, level=logging.INFO)
logger = logging.getLogger(__name__)

WINDOW_WIDTH: int = 1000
WINDOW_HEIGHT: int = 700
CANVAS_WIDTH: int = 1000
CANVAS_HEIGHT: int = 650
CAM_RESOLUTION: Tuple[int, int] = (640, 480)
PREVIEW_RESOLUTION: Tuple[int, int] = (400, 300)
PREVIEW_POSITION: Tuple[int, int] = (10, 10)
TRASH_SIZE: Tuple[int, int] = (150, 150)
BIN_SIZE: int = 100
DRAG_THRESHOLD: int = 50
ANIMATION_STEPS: int = 30
ANIMATION_DELAY: float = 0.02
DOT_RADIUS: int = 5
DOT_COLOR_BGR: Tuple[int, int, int] = (0, 0, 255)

RECYCLING_INSTRUCTION: Dict[str, str] = {
    "plastic": "Plastics: Empty contents and dispose.",
    "paper": "Paper: Remove stickers and tape.",
    "can": "Can: Rinse with water and crush."
}
ICON_PATHS: Dict[str, str] = {
    "plastic": "플라스틱.png",
    "paper": "종이.png",
    "can": "캔.png"
}

mp_hands = mp.solutions.hands
HAND_DETECTOR = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_index_finger_position(frame: "cv2.Mat") -> Optional[Tuple[int, int]]:
    """
    Return the (x, y) coordinates of the index fingertip in the frame,
    or None if no hand is detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = HAND_DETECTOR.process(rgb)
    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0].landmark[8]
    h, w, _ = frame.shape
    return int(lm.x * w), int(lm.y * h)

class TrashDragApp:
    def __init__(
        self,
        root: Tk,
        image_path: Path,
        trash_type: str,
        camera_index: int = 0
    ) -> None:
        self.root = root
        self.root.title("Recycling Classification Detector")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")

        self.canvas = Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
        self.canvas.pack()

        self.trash_type = trash_type
        self.camera_index = camera_index

        self.trash_image = self._load_image(image_path, TRASH_SIZE)
        self.trash_id = self.canvas.create_image(450, 50, anchor=NW, image=self.trash_image)
        self.trash_bbox = (450, 50, 450 + TRASH_SIZE[0], 50 + TRASH_SIZE[1])

        self.bin_positions = {"plastic": (200, 550), "can": (500, 550), "paper": (800, 550)}
        self.bin_icons: Dict[str, ImageTk.PhotoImage] = {}
        self._draw_bins()
        self._create_instruction_label()

        self.dragging = False
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _load_image(self, path: Path, size: Tuple[int, int]) -> ImageTk.PhotoImage:
        if not path.is_file():
            logger.error("Missing image: %s", path)
            raise FileNotFoundError(path)
        img = Image.open(path).resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _draw_bins(self) -> None:
        for btype, (bx, by) in self.bin_positions.items():
            self.canvas.create_rectangle(
                bx - BIN_SIZE/2, by - BIN_SIZE/2,
                bx + BIN_SIZE/2, by + BIN_SIZE/2,
                fill="lightgray", outline="black"
            )
            self.canvas.create_text(
                bx, by + BIN_SIZE/2 + 20,
                text=btype, font=("Arial", 12, "bold")
            )
            # bin icon
            icon_file = ICON_PATHS.get(btype)
            if icon_file and os.path.exists(icon_file):
                pil_ic = Image.open(icon_file).resize((80, 80), Image.LANCZOS)
                tk_ic = ImageTk.PhotoImage(pil_ic)
                self.bin_icons[btype] = tk_ic
                self.canvas.create_image(bx, by, image=tk_ic)

    def _create_instruction_label(self) -> None:
        inst = RECYCLING_INSTRUCTION.get(self.trash_type, "Info not found.")
        self.instr_lbl = Label(
            self.root,
            text=f"Type: {self.trash_type}\n{inst}",
            font=("Arial", 15), fg="black"
        )
        self.instr_lbl.place(x=10, y=CANVAS_HEIGHT + 10)

    def _camera_loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened(): raise RuntimeError("Camera open failed")
        preview_id: Optional[int] = None

        while True:
            ret, frame = cap.read()
            if not ret: continue

            pos = get_index_finger_position(frame)
            small = cv2.resize(frame, PREVIEW_RESOLUTION)
            if pos:
                sx = int(pos[0] * PREVIEW_RESOLUTION[0] / CAM_RESOLUTION[0])
                sy = int(pos[1] * PREVIEW_RESOLUTION[1] / CAM_RESOLUTION[1])
                cv2.circle(small, (sx, sy), DOT_RADIUS, DOT_COLOR_BGR, -1)

            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil = ImageTk.PhotoImage(Image.fromarray(rgb_small))

            if preview_id is None:
                preview_id = self.canvas.create_image(*PREVIEW_POSITION, anchor=NW, image=pil)
            else:
                self.canvas.itemconfig(preview_id, image=pil)

            if pos:
                fx, fy = self._coords_full(pos)
                if not self.dragging and self._over_trash(fx, fy):
                    self.dragging = True
                if self.dragging:
                    self._drag(fx, fy)
                    if self._try_drop(fx, fy): break

            time.sleep(ANIMATION_DELAY)

        cap.release()

    def _coords_full(self, pos: Tuple[int,int]) -> Tuple[int,int]:
        fx = int(pos[0] * WINDOW_WIDTH / CAM_RESOLUTION[0])
        fy = int(pos[1] * CANVAS_HEIGHT / CAM_RESOLUTION[1])
        return fx, fy

    def _over_trash(self, x:int,y:int) -> bool:
        x1,y1,x2,y2 = self.trash_bbox
        return x1<x<x2 and y1<y<y2

    def _drag(self, x:int,y:int) -> None:
        nx,ny = x-TRASH_SIZE[0]//2, y-TRASH_SIZE[1]//2
        self.canvas.coords(self.trash_id, nx, ny)
        self.trash_bbox = (nx,ny,nx+TRASH_SIZE[0],ny+TRASH_SIZE[1])

    def _try_drop(self, x:int,y:int) -> bool:
        for btype,(bx,by) in self.bin_positions.items():
            if abs(x-bx)<DRAG_THRESHOLD and abs(y-by)<DRAG_THRESHOLD:
                correct = (btype==self.trash_type)
                self._finish_drop(btype,correct)
                return True
        return False

    def _finish_drop(self, btype:str, correct:bool) -> None:
        if correct:
            self._animate(btype)
            msg = f"Correct in {btype}!"
        else:
            msg = "Wrong bin. Please try again."
            self.canvas.coords(self.trash_id, 450,50)
        self.instr_lbl.config(text=msg)
        time.sleep(1)
        self.dragging=False

    def _animate(self, btype:str) -> None:
        sx,sy,*_ = self.trash_bbox
        tx,ty = self.bin_positions[btype]
        ex,ey = tx-TRASH_SIZE[0]//2, ty-TRASH_SIZE[1]//2
        dx,dy = (ex-sx)/ANIMATION_STEPS, (ey-sy)/ANIMATION_STEPS
        for _ in range(ANIMATION_STEPS):
            self.canvas.move(self.trash_id, dx, dy)
            self.root.update_idletasks(); time.sleep(ANIMATION_DELAY)
        self.trash_bbox=(ex,ey,ex+TRASH_SIZE[0],ey+TRASH_SIZE[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("current_trash.jpg"))
    parser.add_argument("--type", type=str, default="trash_type.txt")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not args.image.is_file() or not Path(args.type).is_file():
        raise FileNotFoundError("Missing files")
    with open(args.type, encoding="utf-8") as f:
        ttype = f.read().strip()

    root = Tk()
    TrashDragApp(root, args.image, ttype, camera_index=args.camera)
    root.mainloop()

if __name__ == "__main__": main()
