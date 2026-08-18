# camera.py
#
# OpenCV 기반 USB 카메라 캡처 래퍼. lerobot은 opencv/realsense/zmq 등 여러 백엔드를
# 추상화하지만, 여기서는 이 프로젝트에서 실제 쓰는 opencv 카메라 하나만 지원한다.

import cv2
import numpy as np


class Camera:
    def __init__(self, index_or_path: int | str, width: int = 640, height: int = 480, fps: int = 30):
        self.index_or_path = index_or_path
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None

    def connect(self) -> None:
        self.cap = cv2.VideoCapture(self.index_or_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise OSError(f"카메라를 열 수 없습니다: {self.index_or_path}")

    def read(self) -> np.ndarray:
        """RGB (H, W, 3) uint8 배열을 반환한다."""
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError(f"카메라 프레임을 읽지 못했습니다: {self.index_or_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def disconnect(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
