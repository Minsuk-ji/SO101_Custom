# normalization.py
#
# raw 엔코더 값(Present_Position/Goal_Position, 0~4095, 캘리브레이션의 homing_offset이
# 이미 적용된 값) <-> 정규화된 값 사이의 변환.
#
# 리더와 팔로워는 조립 편차 때문에 캘리브레이션 값(range_min/max)이 서로 다르다.
# 그래서 "리더의 raw 값을 팔로워에 그대로 복사"하면 안 되고, 반드시
# raw(리더) -> 정규화(-100~100, 관절 공통 스케일) -> raw(팔로워) 순으로 변환해야
# 두 팔의 "가동범위 대비 몇 %인지"가 일치하게 된다.
#
# 원본 출처: lerobot/motors/motors_bus.py (_normalize / _unnormalize)
# gripper만 0~100 (닫힘~열림), 나머지 5개 관절은 -100~100 (중립 기준 좌우 대칭)을 쓴다
# — lerobot의 MotorNormMode.RANGE_0_100 / RANGE_M100_100과 동일한 규칙.

GRIPPER_JOINT = "gripper"


def normalize(raw: dict[str, int], calibration: dict[str, dict]) -> dict[str, float]:
    normalized = {}
    for motor, raw_pos in raw.items():
        min_ = calibration[motor]["range_min"]
        max_ = calibration[motor]["range_max"]
        bounded = min(max_, max(min_, raw_pos))  # 가동범위를 벗어난 값은 clip

        if motor == GRIPPER_JOINT:
            normalized[motor] = ((bounded - min_) / (max_ - min_)) * 100
        else:
            normalized[motor] = (((bounded - min_) / (max_ - min_)) * 200) - 100

    return normalized


def unnormalize(normalized: dict[str, float], calibration: dict[str, dict]) -> dict[str, int]:
    raw = {}
    for motor, val in normalized.items():
        min_ = calibration[motor]["range_min"]
        max_ = calibration[motor]["range_max"]

        if motor == GRIPPER_JOINT:
            bounded = min(100.0, max(0.0, val))
            raw[motor] = int((bounded / 100) * (max_ - min_) + min_)
        else:
            bounded = min(100.0, max(-100.0, val))
            raw[motor] = int(((bounded + 100) / 200) * (max_ - min_) + min_)

    return raw


def clip_relative(goal: dict[str, float], current: dict[str, float], max_delta: float) -> dict[str, float]:
    """goal이 current에서 max_delta(정규화 단위) 이상 벗어나지 않도록 한 스텝 이동량을 제한한다.

    선택적 안전장치 — 텔레옵 시작 순간 리더/팔로워 자세가 크게 다르면, 팔로워가 그 차이를
    한 스텝에 급격히 따라잡으려다 다치거나 부품이 상할 수 있다.
    """
    clipped = {}
    for motor, goal_val in goal.items():
        diff = goal_val - current[motor]
        diff = max(-max_delta, min(max_delta, diff))
        clipped[motor] = current[motor] + diff
    return clipped
