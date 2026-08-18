# dataset_writer.py
#
# 로컬 디스크에 에피소드 단위로 텔레옵 데이터를 저장한다. lerobot의 LeRobotDataset은
# 여러 에피소드를 chunk 단위 parquet/mp4로 묶어 저장하는데(대규모 데이터셋 최적화),
# 여기서는 그 복잡도 없이 "에피소드 1개 = 폴더 1개"로 단순화했다 — 이해하기 쉽고,
# 개인 프로젝트 규모(수십~수백 에피소드)에서는 굳이 chunk로 합칠 이유가 없다.
#
# 디스크 레이아웃:
#   <root>/
#     info.json                 # fps, 관절 이름 순서, 카메라 이름, task, 에피소드 수
#     episode_000000/
#       frames.npz               # observation.state[T,D], action[T,D], timestamp[T]
#       videos/<camera>.mp4
#     episode_000001/
#       ...

import json
from pathlib import Path

import cv2
import numpy as np

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")  # 별도 코덱 설치 없이 대부분의 opencv 빌드에서 동작


class DatasetWriter:
    def __init__(
        self,
        root: str | Path,
        fps: int,
        joint_names: list[str],
        camera_names: list[str],
        task: str,
    ):
        self.root = Path(root)
        self.fps = fps
        self.joint_names = joint_names
        self.camera_names = camera_names
        self.task = task

        self.root.mkdir(parents=True, exist_ok=True)
        self._info_path = self.root / "info.json"
        self._episode_count = self._load_existing_episode_count()

        self._episode_dir: Path | None = None
        self._states: list[list[float]] = []
        self._actions: list[list[float]] = []
        self._timestamps: list[float] = []
        self._video_writers: dict[str, cv2.VideoWriter] = {}

    def _load_existing_episode_count(self) -> int:
        if self._info_path.is_file():
            return json.loads(self._info_path.read_text())["num_episodes"]
        return 0

    @property
    def num_episodes(self) -> int:
        return self._episode_count

    def start_episode(self) -> None:
        self._episode_dir = self.root / f"episode_{self._episode_count:06d}"
        (self._episode_dir / "videos").mkdir(parents=True, exist_ok=True)
        self._states, self._actions, self._timestamps = [], [], []

        # 해상도는 첫 프레임이 들어올 때 알 수 있으므로, VideoWriter는 add_frame에서 지연 생성한다.
        self._video_writers = dict.fromkeys(self.camera_names)

    def add_frame(
        self,
        state: dict[str, float],
        action: dict[str, float],
        images: dict[str, np.ndarray],
        timestamp: float,
    ) -> None:
        if self._episode_dir is None:
            raise RuntimeError("start_episode()를 먼저 호출하세요.")

        self._states.append([state[name] for name in self.joint_names])
        self._actions.append([action[name] for name in self.joint_names])
        self._timestamps.append(timestamp)

        for cam, image_rgb in images.items():
            writer = self._video_writers.get(cam)
            if writer is None:
                h, w = image_rgb.shape[:2]
                path = self._episode_dir / "videos" / f"{cam}.mp4"
                writer = cv2.VideoWriter(str(path), FOURCC, self.fps, (w, h))
                self._video_writers[cam] = writer
            writer.write(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    def save_episode(self) -> Path:
        if self._episode_dir is None:
            raise RuntimeError("start_episode()를 먼저 호출하세요.")

        for writer in self._video_writers.values():
            if writer is not None:
                writer.release()

        np.savez(
            self._episode_dir / "frames.npz",
            **{"observation.state": np.array(self._states, dtype=np.float32)},
            **{"action": np.array(self._actions, dtype=np.float32)},
            timestamp=np.array(self._timestamps, dtype=np.float64),
        )

        saved_dir = self._episode_dir
        self._episode_count += 1
        self._write_info()
        self._episode_dir = None
        return saved_dir

    def discard_episode(self) -> None:
        """재촬영(re-record) 시 지금 채우고 있던 에피소드를 버린다."""
        for writer in self._video_writers.values():
            if writer is not None:
                writer.release()
        if self._episode_dir is not None:
            for f in self._episode_dir.rglob("*"):
                if f.is_file():
                    f.unlink()
        self._episode_dir = None

    def _write_info(self) -> None:
        info = {
            "fps": self.fps,
            "joint_names": self.joint_names,
            "camera_names": self.camera_names,
            "task": self.task,
            "num_episodes": self._episode_count,
        }
        self._info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
