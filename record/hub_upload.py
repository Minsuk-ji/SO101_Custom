# hub_upload.py
#
# 로컬 데이터셋 디렉터리를 통째로 HuggingFace Hub의 dataset repo로 업로드한다.
# lerobot의 LeRobotDataset.push_to_hub()도 결국 내부적으로 huggingface_hub의
# upload_folder를 그대로 호출하는 것과 동일한 방식이다 — 우리는 그 한 걸음을
# lerobot 없이 huggingface_hub 라이브러리만으로 직접 수행한다.

from pathlib import Path

from huggingface_hub import HfApi


def push_dataset_to_hub(root: str | Path, repo_id: str, private: bool = True) -> str:
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(root))
    return f"https://huggingface.co/datasets/{repo_id}"
