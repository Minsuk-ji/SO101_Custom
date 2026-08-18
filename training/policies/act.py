# act.py
#
# ACT(Action Chunking Transformer)를 뼈대만 남겨 다시 구현한 것. 원본
# (lerobot/policies/act/modeling_act.py, ~750줄)의 핵심 아이디어 3가지를 그대로 따른다:
#
#  1. 매 스텝 하나씩이 아니라, 앞으로 chunk_size 스텝만큼의 행동을 한 번에 예측한다
#     (사람이 리더암을 조작할 때 생기는 떨림/비일관성에 덜 민감해짐).
#  2. CVAE: 학습 시에는 "실제로 무슨 행동을 했는지"(정답 action chunk)까지 인코딩해서
#     스타일 잠재변수 z를 뽑아 디코더에 조건으로 준다. 추론 시에는 z=0(사전분포 평균)을
#     쓴다 — 같은 관측이라도 사람마다 다르게 움직였을 수 있는 걸 z가 흡수하게 학습된다.
#  3. 트랜스포머 인코더(카메라 토큰+상태+z) -> 디코더(학습된 쿼리 chunk_size개가
#     인코더 출력을 cross-attention으로 참조) -> 각 쿼리 위치를 행동으로 projection.
#
# torch.nn의 TransformerEncoder/Decoder를 그대로 사용해 attention 자체를 다시
# 구현하지는 않았다 — 여기서 다시 만들 가치가 있는 부분은 "청크+CVAE+인코더-디코더를
# 어떻게 로봇 조작에 연결하는가"이지, attention 커널 자체가 아니기 때문이다.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..policy_base import Policy
from ..registry import register_policy


class ImageEncoder(nn.Module):
    """카메라 이미지 한 장을 고정 길이 벡터로 뭉개는 작은 CNN (mlp_bc가 사용)."""

    def __init__(self, out_dim: int = 64, in_size: int = 96):
        super().__init__()
        self.in_size = in_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)
        x = self.conv(x).mean(dim=(2, 3))  # global average pool
        return self.fc(x)


class ImageBackbone(nn.Module):
    """카메라 이미지 한 장을 공간 토큰 시퀀스로 바꾸는 CNN (ACT가 사용).

    풀링해서 벡터 하나로 뭉개는 ImageEncoder와 달리, 공간 위치별 특징을
    각각 하나의 토큰으로 남겨서 트랜스포머가 "어디에 뭐가 있는지"를 attention으로
    골라볼 수 있게 한다.
    """

    def __init__(self, d_model: int, in_size: int = 96):
        super().__init__()
        self.in_size = in_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, d_model, 3, stride=2, padding=1),
            nn.ReLU(),
        )  # in_size=96 -> 16배 다운샘플 -> 6x6 = 36 토큰

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.in_size, self.in_size), mode="bilinear", align_corners=False)
        feat = self.conv(x)  # (B, d_model, H', W')
        return feat.flatten(2).transpose(1, 2)  # (B, H'*W', d_model)


class CVAEEncoder(nn.Module):
    """학습 시에만 쓰인다: (상태, 정답 action chunk) -> 잠재변수 z의 평균/분산."""

    def __init__(self, action_dim: int, state_dim: int, d_model: int, latent_dim: int, chunk_size: int, n_heads: int):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, chunk_size + 2, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.to_stats = nn.Linear(d_model, latent_dim * 2)

    def forward(self, action_chunk: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = action_chunk.shape[0]
        tokens = torch.cat(
            [self.cls_token.expand(b, -1, -1), self.state_proj(state).unsqueeze(1), self.action_proj(action_chunk)],
            dim=1,
        )
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        cls_out = self.encoder(tokens)[:, 0]
        mu, logvar = self.to_stats(cls_out).chunk(2, dim=-1)
        return mu, logvar


@register_policy("act")
class ACTPolicy(Policy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        camera_names: list[str],
        chunk_size: int = 32,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        image_size: int = 96,
    ):
        super().__init__()
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.image_backbones = nn.ModuleDict(
            {cam: ImageBackbone(d_model, image_size) for cam in camera_names}
        )
        self.state_proj = nn.Linear(state_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.cvae_encoder = CVAEEncoder(action_dim, state_dim, d_model, latent_dim, chunk_size, n_heads)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=d_model * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)
        self.query_embed = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)

        self.action_head = nn.Linear(d_model, action_dim)

        self._action_queue: list[torch.Tensor] = []

    def _predict_chunk(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        tokens = [self.latent_proj(z).unsqueeze(1), self.state_proj(batch["observation.state"]).unsqueeze(1)]
        for cam in self.camera_names:
            tokens.append(self.image_backbones[cam](batch[f"observation.images.{cam}"]))
        memory = self.encoder(torch.cat(tokens, dim=1))

        b = memory.shape[0]
        queries = self.query_embed.expand(b, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)  # (B, chunk_size, action_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        target = batch["action"][:, : self.chunk_size]
        mu, logvar = self.cvae_encoder(target, batch["observation.state"])
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)  # reparameterization trick

        pred = self._predict_chunk(batch, z)
        recon_loss = F.l1_loss(pred, target)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.kl_weight * kl_loss
        return loss, {"l1": recon_loss.item(), "kl": kl_loss.item()}

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self._action_queue:
            with torch.no_grad():
                b = batch["observation.state"].shape[0]
                z = torch.zeros(b, self.latent_dim, device=batch["observation.state"].device)
                chunk = self._predict_chunk(batch, z)
            self._action_queue = list(chunk.unbind(dim=1))  # chunk_size개의 (B, action_dim) 텐서
        return self._action_queue.pop(0)

    def reset(self) -> None:
        self._action_queue = []
