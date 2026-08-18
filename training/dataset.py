# dataset.py
#
# record/ 모듈이 저장한 에피소드들을 PyTorch Dataset으로 읽는다. 각 샘플은
# 한 시점의 관측(상태+카메라 이미지)과, 그 시점부터 chunk_size 스텝만큼의
# 정답 행동 시퀀스로 구성된다 (ACT 같은 청크 예측 정책과 mlp_bc 둘 다 이 형식을 쓴다 —
# chunk_size=1로 두면 mlp_bc가 그냥 한 스텝만 쓰는 것과 동일해진다).
#
# 에피소드가 짧아서(수백 프레임) 비디오를 프레임 단위로 전부 디코딩해 메모리에
# 캐시해둔다 — lerobot처럼 매번 디스크에서 seek해서 읽는 것보다 훨씬 단순하고,
# 개인 프로젝트 규모(에피소드 수십 개)에서는 메모리 문제도 없다.

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


class SO101Dataset(Dataset):
    def __init__(self, root: str | Path, chunk_size: int = 32):
        self.root = Path(root)
        info = json.loads((self.root / "info.json").read_text())
        self.joint_names: list[str] = info["joint_names"]
        self.camera_names: list[str] = info["camera_names"]
        self.fps: int = info["fps"]
        self.task: str = info["task"]
        self.chunk_size = chunk_size

        self._episode_cache: dict[int, dict] = {}
        self.index: list[tuple[int, int]] = []
        for ep in range(info["num_episodes"]):
            data = self._load_episode(ep)
            for t in range(data["state"].shape[0]):
                self.index.append((ep, t))

    def _load_episode(self, ep_idx: int) -> dict:
        if ep_idx in self._episode_cache:
            return self._episode_cache[ep_idx]

        ep_dir = self.root / f"episode_{ep_idx:06d}"
        npz = np.load(ep_dir / "frames.npz")
        images = {cam: _decode_video(ep_dir / "videos" / f"{cam}.mp4") for cam in self.camera_names}
        data = {"state": npz["observation.state"], "action": npz["action"], "images": images}
        self._episode_cache[ep_idx] = data
        return data

    @property
    def state_dim(self) -> int:
        return len(self.joint_names)

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep, t = self.index[idx]
        data = self._load_episode(ep)
        episode_len = data["state"].shape[0]

        end = min(t + self.chunk_size, episode_len)
        action_chunk = data["action"][t:end]
        if len(action_chunk) < self.chunk_size:
            # 에피소드 끝에서는 마지막 행동을 반복해 chunk_size를 채운다
            # (팔이 마지막 자세에서 정지해 있었다고 보는 것과 같다).
            pad = np.repeat(action_chunk[-1:], self.chunk_size - len(action_chunk), axis=0)
            action_chunk = np.concatenate([action_chunk, pad], axis=0)

        sample = {
            "observation.state": torch.from_numpy(data["state"][t]).float(),
            "action": torch.from_numpy(action_chunk).float(),
        }
        for cam, frames in data["images"].items():
            img = frames[min(t, frames.shape[0] - 1)]  # (H, W, 3) uint8
            sample[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return sample
