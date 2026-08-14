# teleoperate.py
#
# 리더암을 손으로 움직이면 팔로워암이 그대로 따라 움직이게 하는 실시간 루프.
# lerobot의 lerobot_teleoperate.py(카메라/시각화/프로세서 파이프라인 포함)를
# 그 핵심 루프만 남기고 다시 구현한 것.
#
# 절차 (매 프레임):
#   1. 리더의 raw 위치 6개를 읽는다
#   2. 리더 자신의 캘리브레이션으로 정규화 (-100~100 / gripper는 0~100)
#   3. (옵션) 직전 팔로워 목표치에서 너무 크게 벗어나지 않도록 제한
#   4. 팔로워 자신의 캘리브레이션으로 역정규화 -> 팔로워의 raw 목표 위치로 변환
#   5. 팔로워에 Goal_Position으로 write
#
# 정규화를 한 번 거치는 이유: 리더/팔로워는 조립 편차 때문에 range_min/max가 서로
# 다르다 (normalization.py 참고). raw 값을 그대로 복사하면 팔로워가 엉뚱한 각도로
# 움직인다.
#
# 실행 (repo 루트에서, calibration.calibrate로 leader/follower 둘 다 캘리브레이션 완료된 상태여야 함):
#   conda run -n lerobot python -m teleoperate.teleoperate \
#       --leader-port /dev/ttyACM1 --leader-name leader \
#       --follower-port /dev/ttyACM0 --follower-name follower

import argparse
import time

from calibration.calibration_io import load_calibration
from calibration.sign_magnitude import decode_sign_magnitude, encode_sign_magnitude
from motor_setup.feetech_bus import FeetechBus
from motor_setup.sts3215_table import (
    DEFAULT_BAUDRATE,
    GOAL_POSITION,
    OPERATING_MODE,
    PRESENT_POSITION,
    PRESENT_POSITION_SIGN_BIT,
    SO101_MOTOR_IDS,
    TORQUE_ENABLE,
)

from .normalization import clip_relative, normalize, unnormalize


def read_positions(bus: FeetechBus) -> dict[str, int]:
    addr, length = PRESENT_POSITION
    return {
        name: decode_sign_magnitude(bus.read(addr, length, motor_id), PRESENT_POSITION_SIGN_BIT)
        for name, motor_id in SO101_MOTOR_IDS.items()
    }


def write_goal_positions(bus: FeetechBus, raw_goals: dict[str, int]) -> None:
    addr, length = GOAL_POSITION
    for name, motor_id in SO101_MOTOR_IDS.items():
        encoded = encode_sign_magnitude(raw_goals[name], PRESENT_POSITION_SIGN_BIT)
        bus.write(addr, length, motor_id, encoded)


def teleoperate(
    leader_port: str,
    follower_port: str,
    leader_name: str,
    follower_name: str,
    fps: int = 30,
    max_relative_step: float | None = None,
) -> None:
    leader_cal = load_calibration(leader_name)
    follower_cal = load_calibration(follower_name)

    leader = FeetechBus(leader_port)
    follower = FeetechBus(follower_port)
    leader.connect()
    follower.connect()
    leader.set_baudrate(DEFAULT_BAUDRATE)
    follower.set_baudrate(DEFAULT_BAUDRATE)

    torque_addr, torque_len = TORQUE_ENABLE
    mode_addr, mode_len = OPERATING_MODE

    try:
        for motor_id in SO101_MOTOR_IDS.values():
            leader.write(torque_addr, torque_len, motor_id, 0)  # 사람이 손으로 움직여야 하므로 OFF
            follower.write(mode_addr, mode_len, motor_id, 0)  # 0 = position mode
            follower.write(torque_addr, torque_len, motor_id, 1)  # 실제로 움직여야 하므로 ON

        print(f"텔레옵 시작 (fps={fps}). 리더암을 움직여 보세요. Ctrl+C로 종료.")
        period = 1.0 / fps
        last_goal_norm = normalize(read_positions(follower), follower_cal)

        while True:
            loop_start = time.perf_counter()

            leader_raw = read_positions(leader)
            action_norm = normalize(leader_raw, leader_cal)

            if max_relative_step is not None:
                action_norm = clip_relative(action_norm, last_goal_norm, max_relative_step)
            last_goal_norm = action_norm

            follower_raw_goal = unnormalize(action_norm, follower_cal)
            write_goal_positions(follower, follower_raw_goal)

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        for motor_id in SO101_MOTOR_IDS.values():
            follower.write(torque_addr, torque_len, motor_id, 0)
        leader.disconnect()
        follower.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--leader-name", required=True, help="calibration/data/<name>.json")
    parser.add_argument("--follower-name", required=True, help="calibration/data/<name>.json")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--max-relative-step",
        type=float,
        default=None,
        help="한 프레임당 정규화 단위(-100~100 스케일)로 이동 가능한 최대량. 지정 안 하면 제한 없음.",
    )
    args = parser.parse_args()
    teleoperate(
        args.leader_port,
        args.follower_port,
        args.leader_name,
        args.follower_name,
        fps=args.fps,
        max_relative_step=args.max_relative_step,
    )
