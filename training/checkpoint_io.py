# checkpoint_io.py
#
# 체크포인트를 로컬 디렉터리에 저장하거나(항상), 선택적으로 HuggingFace Hub 모델
# repo에도 업로드한다. record/hub_upload.py와 동일한 방식 — huggingface_hub의
# upload_folder/snapshot_download를 직접 쓴다.

import json
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "model.pt"


def save_checkpoint(model: torch.nn.Module, config: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / WEIGHTS_FILENAME)
    (out_dir / CONFIG_FILENAME).write_text(json.dumps(config, indent=2, ensure_ascii=False))
    return out_dir


def load_checkpoint(model: torch.nn.Module, checkpoint_dir: str | Path, map_location: str = "cpu") -> None:
    checkpoint_dir = Path(checkpoint_dir)
    state_dict = torch.load(checkpoint_dir / WEIGHTS_FILENAME, map_location=map_location)
    model.load_state_dict(state_dict)


def load_config(checkpoint_dir: str | Path) -> dict:
    return json.loads((Path(checkpoint_dir) / CONFIG_FILENAME).read_text())


def push_checkpoint_to_hub(checkpoint_dir: str | Path, repo_id: str, private: bool = True) -> str:
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(checkpoint_dir))
    return f"https://huggingface.co/{repo_id}"


def pull_checkpoint_from_hub(repo_id: str, local_dir: str | Path) -> Path:
    path = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(local_dir))
    return Path(path)
