# train.py
#
# 정책 종류에 무관한 범용 학습 루프. record/로 모은 데이터셋을 읽어 --policy로
# 지정한 정책(mlp_bc, act, ... training/policies/에 등록된 것 아무거나)을 학습한다.
#
# 새 정책을 추가해도 이 파일은 건드릴 필요가 없다 — Policy 인터페이스
# (forward/select_action)만 구현하고 @register_policy만 붙이면 --policy=<이름>으로
# 바로 쓸 수 있다 (registry.py 참고).
#
# 실행 (repo 루트에서):
#   conda run -n lerobot python -m training.train \
#       --policy act --dataset ~/so101_data/pick_cube \
#       --checkpoint-dir ~/so101_checkpoints/pick_cube_act \
#       --epochs 100 --batch-size 8
#
#   conda run -n lerobot python -m training.train \
#       --policy mlp_bc --dataset ~/so101_data/pick_cube \
#       --checkpoint-dir ~/so101_checkpoints/pick_cube_mlp \
#       --push-to-hub --repo-id myuser/so101-pick-cube-mlp

import argparse

import torch
from torch.utils.data import DataLoader

from . import policies  # noqa: F401 — import해야 각 정책이 registry에 등록된다
from .checkpoint_io import push_checkpoint_to_hub, save_checkpoint
from .dataset import SO101Dataset
from .registry import POLICY_REGISTRY, get_policy_class


def build_policy(policy_name: str, dataset: SO101Dataset, **overrides):
    cls = get_policy_class(policy_name)
    kwargs = dict(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        camera_names=dataset.camera_names,
        chunk_size=dataset.chunk_size,
    )
    kwargs.update(overrides)
    return cls(**kwargs), kwargs


def train(
    policy_name: str,
    dataset_root: str,
    checkpoint_dir: str,
    chunk_size: int = 32,
    batch_size: int = 8,
    epochs: int = 100,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    push_to_hub: bool = False,
    repo_id: str | None = None,
    private: bool = True,
    log_every: int = 20,
) -> None:
    dataset = SO101Dataset(dataset_root, chunk_size=chunk_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"데이터셋: {len(dataset)} 프레임, 관절 {dataset.joint_names}, 카메라 {dataset.camera_names}")

    policy, policy_kwargs = build_policy(policy_name, dataset)
    policy.to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logs = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_every == 0:
                print(f"epoch={epoch:4d} step={step:6d} loss={loss.item():.4f} {logs}")
            step += 1

    config = {
        "policy": policy_name,
        "policy_kwargs": policy_kwargs,
        "joint_names": dataset.joint_names,
        "camera_names": dataset.camera_names,
        "task": dataset.task,
    }
    saved_dir = save_checkpoint(policy, config, checkpoint_dir)
    print(f"체크포인트 저장: {saved_dir}")

    if push_to_hub:
        if not repo_id:
            raise ValueError("--push-to-hub 사용 시 --repo-id가 필요합니다.")
        url = push_checkpoint_to_hub(saved_dir, repo_id, private=private)
        print(f"HuggingFace Hub 업로드 완료: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, choices=list(POLICY_REGISTRY))
    parser.add_argument("--dataset", required=True, help="record/로 저장한 데이터셋 루트 경로")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default=None, help="예: myuser/so101-pick-cube-act")
    parser.add_argument("--private", action="store_true", default=True)
    args = parser.parse_args()

    train(
        policy_name=args.policy,
        dataset_root=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private,
    )
