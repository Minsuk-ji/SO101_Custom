# mlp_bc.py
#
# 가장 단순한 baseline: 작은 CNN으로 카메라 이미지를 특징 벡터로 뭉개고, 관절 상태와
# 합쳐서 MLP로 바로 행동을 예측한다. 청크(여러 스텝 묶음) 예측이나 CVAE 같은
# ACT의 장치가 전혀 없다 — "일단 파이프라인이 도는지" 빠르게 확인하거나, 다른
# 정책의 성능을 비교할 기준선으로 쓴다.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..policy_base import Policy
from ..registry import register_policy
from .act import ImageEncoder


@register_policy("mlp_bc")
class MLPBCPolicy(Policy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        camera_names: list[str],
        chunk_size: int = 1,
        hidden_dim: int = 256,
        image_feat_dim: int = 64,
    ):
        super().__init__()
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.image_encoders = nn.ModuleDict(
            {cam: ImageEncoder(out_dim=image_feat_dim) for cam in camera_names}
        )
        in_dim = state_dim + image_feat_dim * len(camera_names)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * chunk_size),
        )

    def _predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feats = [batch["observation.state"]]
        for cam in self.camera_names:
            feats.append(self.image_encoders[cam](batch[f"observation.images.{cam}"]))
        x = torch.cat(feats, dim=-1)
        return self.mlp(x).view(-1, self.chunk_size, self.action_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        pred = self._predict(batch)
        target = batch["action"][:, : self.chunk_size]
        loss = F.mse_loss(pred, target)
        return loss, {"mse": loss.item()}

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self._predict(batch)[:, 0]

    def reset(self) -> None:
        pass
