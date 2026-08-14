# apply_calibration.py
#
# 저장된 캘리브레이션 JSON을 읽어 모터 EPROM(Homing_Offset, Min/Max Position Limit)에
# 다시 써준다. 매번 손으로 재-캘리브레이션하지 않고, 이미 구한 값을 다른 세션에서
# (혹은 전원을 새로 켠 모터에) 그대로 복원/검증하고 싶을 때 사용한다.
#
# 실행 (repo 루트에서):
#   conda run -n lerobot python -m calibration.apply_calibration --port /dev/ttyACM1 --name leader

import argparse

from motor_setup.feetech_bus import FeetechBus
from motor_setup.sts3215_table import (
    DEFAULT_BAUDRATE,
    HOMING_OFFSET,
    HOMING_OFFSET_SIGN_BIT,
    MAX_POSITION_LIMIT,
    MIN_POSITION_LIMIT,
    PRESENT_POSITION,
    PRESENT_POSITION_SIGN_BIT,
)

from .calibration_io import load_calibration
from .sign_magnitude import decode_sign_magnitude, encode_sign_magnitude


def apply_calibration(port: str, name: str) -> None:
    calibration = load_calibration(name)

    bus = FeetechBus(port)
    bus.connect()
    bus.set_baudrate(DEFAULT_BAUDRATE)

    try:
        homing_addr, homing_len = HOMING_OFFSET
        min_addr, min_len = MIN_POSITION_LIMIT
        max_addr, max_len = MAX_POSITION_LIMIT
        pos_addr, pos_len = PRESENT_POSITION

        print(f"{'joint':<16}{'id':>4}{'homing':>9}{'min':>7}{'max':>7}{'pos_now':>9}")
        for name_, cal in calibration.items():
            motor_id = cal["id"]
            encoded_offset = encode_sign_magnitude(cal["homing_offset"], HOMING_OFFSET_SIGN_BIT)
            bus.write(homing_addr, homing_len, motor_id, encoded_offset)
            bus.write(min_addr, min_len, motor_id, cal["range_min"])
            bus.write(max_addr, max_len, motor_id, cal["range_max"])

            pos_now = decode_sign_magnitude(bus.read(pos_addr, pos_len, motor_id), PRESENT_POSITION_SIGN_BIT)
            print(
                f"{name_:<16}{motor_id:>4}{cal['homing_offset']:>9}"
                f"{cal['range_min']:>7}{cal['range_max']:>7}{pos_now:>9}"
            )
    finally:
        bus.disconnect()

    print("\n캘리브레이션 적용 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="예: /dev/ttyACM0")
    parser.add_argument("--name", required=True, help="불러올 캘리브레이션 파일 이름 (예: leader)")
    args = parser.parse_args()
    apply_calibration(args.port, args.name)
