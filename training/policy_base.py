# policy_base.py
#
# 모든 정책(policy)이 구현해야 하는 공통 인터페이스. lerobot의 PreTrainedPolicy를
# 참고했지만, 여기서는 허브 업로드/정규화 파이프라인 같은 부가 기능을 빼고
# "학습 루프가 정책 종류와 무관하게 동일하게 동작하기 위해 꼭 필요한 것"만 남겼다:
# forward()가 손실을 계산하고, select_action()이 실제 로봇에 보낼 행동을 낸다.
#
# train.py는 이 인터페이스에만 의존하므로, 새 정책을 추가할 때 train.py를
# 전혀 건드릴 필요가 없다 — policies/ 아래에 파일 하나 추가하고 @register_policy만
# 붙이면 --policy=<이름>으로 바로 쓸 수 있다.

import abc

import torch
import torch.nn as nn


class Policy(nn.Module, abc.ABC):
    name: str  # register_policy 데코레이터가 채워준다

    @abc.abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """학습용. batch를 받아 (loss, 로그용 dict)를 반환한다."""

    @abc.abstractmethod
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """추론용. batch(배치 크기 1)를 받아 실제로 로봇에 보낼 행동 [B, action_dim] 하나를 반환한다."""

    def reset(self) -> None:
        """에피소드 시작 시 내부 상태(예: 행동 청크 큐)를 초기화한다. 기본은 아무 것도 안 함."""
