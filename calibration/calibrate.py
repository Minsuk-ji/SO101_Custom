# calibrate.py
#
# SO-101 팔 하나(리더 또는 팔로워)의 캘리브레이션을 진행한다.
# lerobot의 SOLeader.calibrate() / MotorsBus.set_half_turn_homings() /
# record_ranges_of_motion()을 참고해, 이 프로젝트에 맞게 다시 구현한 것.
#
# 절차:
#   1. 모든 모터 Torque 끄기 + 위치 제어 모드로 설정
#   2. 기존 Homing_Offset/Min/Max를 초기화 (0, 0, 4095)
#   3. 사용자가 팔을 가동범위 중앙으로 옮기면, 그 위치를 "절반 회전(2047)"이
#      되도록 Homing_Offset을 계산해서 기록 (원점 보정)
#   4. wrist_roll을 제외한 관절들을 손으로 끝까지 움직이며 min/max encoder 값을 기록
#      (wrist_roll은 360도 자유 회전이라 항상 0~4095 그대로 사용)
#   5. 계산된 값을 모터 EPROM에 쓰고, JSON으로도 저장
#
# 실행 (repo 루트에서):
#   conda run -n lerobot python -m calibration.calibrate --port /dev/ttyACM1 --name leader

import argparse
import select
import sys
import time

from motor_setup.feetech_bus import FeetechBus
from motor_setup.sts3215_table import (
    DEFAULT_BAUDRATE,
    FULL_TURN_MOTOR,
    HOMING_OFFSET,
    HOMING_OFFSET_SIGN_BIT,
    MAX_POSITION_LIMIT,
    MIN_POSITION_LIMIT,
    MODEL_RESOLUTION,
    OPERATING_MODE,
    PRESENT_POSITION,
    PRESENT_POSITION_SIGN_BIT,
    SO101_MOTOR_IDS,
    TORQUE_ENABLE,
)

from .calibration_io import save_calibration
from .sign_magnitude import decode_sign_magnitude, encode_sign_magnitude

HALF_TURN = (MODEL_RESOLUTION - 1) // 2  # 4095 // 2 == 2047


def enter_pressed() -> bool:
    """블로킹 없이 Enter가 눌렸는지 확인 (stdin에 입력이 준비됐는지만 확인)."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(ready) and sys.stdin.readline().strip() == ""


def read_position(bus: FeetechBus, motor_id: int) -> int:
    addr, length = PRESENT_POSITION
    raw = bus.read(addr, length, motor_id)
    return decode_sign_magnitude(raw, PRESENT_POSITION_SIGN_BIT)


def write_homing_offset(bus: FeetechBus, motor_id: int, offset: int) -> None:
    addr, length = HOMING_OFFSET
    bus.write(addr, length, motor_id, encode_sign_magnitude(offset, HOMING_OFFSET_SIGN_BIT))


def reset_calibration(bus: FeetechBus, ids: list[int]) -> None:
    for motor_id in ids:
        write_homing_offset(bus, motor_id, 0)
        min_addr, min_len = MIN_POSITION_LIMIT
        max_addr, max_len = MAX_POSITION_LIMIT
        bus.write(min_addr, min_len, motor_id, 0)
        bus.write(max_addr, max_len, motor_id, MODEL_RESOLUTION - 1)


def set_half_turn_homings(bus: FeetechBus, ids: list[int]) -> dict[int, int]:
    """지금 위치가 Present_Position=HALF_TURN으로 읽히도록 Homing_Offset을 계산해서 쓴다."""
    offsets = {}
    for motor_id in ids:
        raw_pos = read_position(bus, motor_id)  # 이 시점엔 offset=0이라 raw 그대로
        offset = raw_pos - HALF_TURN
        write_homing_offset(bus, motor_id, offset)
        offsets[motor_id] = offset
    return offsets


def record_ranges_of_motion(bus: FeetechBus, id_to_name: dict[int, str]) -> tuple[dict[int, int], dict[int, int]]:
    ids = list(id_to_name)
    positions = {motor_id: read_position(bus, motor_id) for motor_id in ids}
    mins = dict(positions)
    maxes = dict(positions)

    print("\n각 관절을 손으로 끝까지 움직여보세요. 멈추려면 Enter...")
    while not enter_pressed():
        print("\033[F" * (len(ids) + 2), end="")  # 이전 출력 지우고 덮어쓰기
        print(f"{'joint':<16}{'min':>6}{'pos':>6}{'max':>6}")
        for motor_id in ids:
            pos = read_position(bus, motor_id)
            mins[motor_id] = min(mins[motor_id], pos)
            maxes[motor_id] = max(maxes[motor_id], pos)
            print(f"{id_to_name[motor_id]:<16}{mins[motor_id]:>6}{pos:>6}{maxes[motor_id]:>6}")
        time.sleep(0.02)

    return mins, maxes


def calibrate(port: str, name: str) -> None:
    id_to_name = {v: k for k, v in SO101_MOTOR_IDS.items()}
    ids = list(id_to_name)

    bus = FeetechBus(port)
    bus.connect()
    bus.set_baudrate(DEFAULT_BAUDRATE)

    try:
        torque_addr, torque_len = TORQUE_ENABLE
        mode_addr, mode_len = OPERATING_MODE
        for motor_id in ids:
            bus.write(torque_addr, torque_len, motor_id, 0)  # 손으로 움직여야 하므로 torque off
            bus.write(mode_addr, mode_len, motor_id, 0)  # 0 = position mode

        reset_calibration(bus, ids)

        input("\n팔을 가동범위 중앙(대략 중립 자세)으로 이동시킨 뒤 Enter...")
        homing_offsets = set_half_turn_homings(bus, ids)

        moving_ids = [i for i in ids if id_to_name[i] != FULL_TURN_MOTOR]
        mins, maxes = record_ranges_of_motion(bus, {i: id_to_name[i] for i in moving_ids})

        full_turn_id = SO101_MOTOR_IDS[FULL_TURN_MOTOR]
        mins[full_turn_id] = 0
        maxes[full_turn_id] = MODEL_RESOLUTION - 1

        min_addr, min_len = MIN_POSITION_LIMIT
        max_addr, max_len = MAX_POSITION_LIMIT
        calibration = {}
        for motor_id in ids:
            bus.write(min_addr, min_len, motor_id, mins[motor_id])
            bus.write(max_addr, max_len, motor_id, maxes[motor_id])
            calibration[id_to_name[motor_id]] = {
                "id": motor_id,
                "drive_mode": 0,
                "homing_offset": homing_offsets[motor_id],
                "range_min": mins[motor_id],
                "range_max": maxes[motor_id],
            }

        saved_path = save_calibration(name, calibration)
        print(f"\n캘리브레이션 저장 완료: {saved_path}")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="예: /dev/ttyACM0")
    parser.add_argument("--name", required=True, help="저장 파일 이름 (예: leader, follower)")
    args = parser.parse_args()
    calibrate(args.port, args.name)
