# calibration_io.py
#
# 계산된 캘리브레이션 값을 JSON으로 저장/로드한다.
# lerobot의 캘리브레이션 파일과 같은 필드 구성(id, drive_mode, homing_offset,
# range_min, range_max)을 쓰지만, 저장 위치는 lerobot의 HF 캐시 경로가 아니라
# 이 저장소 안(calibration/data/)으로 둔다 — lerobot 없이도 이 프로젝트만으로
# 캘리브레이션을 재현/검증할 수 있게 하기 위함.

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def calibration_path(name: str) -> Path:
    """name 예: 'leader', 'follower'."""
    return DATA_DIR / f"{name}.json"


def save_calibration(name: str, calibration: dict[str, dict]) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    path = calibration_path(name)
    path.write_text(json.dumps(calibration, indent=4, ensure_ascii=False), encoding="utf-8")
    return path


def load_calibration(name: str) -> dict[str, dict]:
    path = calibration_path(name)
    if not path.is_file():
        raise FileNotFoundError(f"캘리브레이션 파일이 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
