# scan_bus.py
#
# 데이지체인으로 완전히 조립된 SO-101 팔 하나를 대상으로, 현재 버스에 응답하는
# 모든 모터의 (id, model_number)를 스캔해서 보여준다.
# setup_motors.py가 이미 끝난 뒤 "실제로 잘 매핑됐는지" 확인하는 용도.
#
# 실행: conda run -n lerobot python -m motor_setup.scan_bus --port /dev/ttyACM1

import argparse

from .feetech_bus import FeetechBus
from .sts3215_table import DEFAULT_BAUDRATE, SO101_MOTOR_IDS


def scan_bus(port: str) -> None:
    bus = FeetechBus(port)
    bus.connect()
    id_to_name = {v: k for k, v in SO101_MOTOR_IDS.items()}

    try:
        # 매핑이 끝났다면 전부 DEFAULT_BAUDRATE에서 응답해야 정상이다.
        bus.set_baudrate(DEFAULT_BAUDRATE)
        print(f"baudrate={DEFAULT_BAUDRATE} 로 ID 1~252 스캔 중...\n")

        found = {}
        for motor_id in range(1, 253):
            model_number = bus.ping(motor_id)
            if model_number is not None:
                found[motor_id] = model_number

        if not found:
            print("응답하는 모터가 없습니다. 연결/전원을 확인하세요.")
            return

        print(f"{'ID':<4}{'관절명':<16}{'model_number':<14}")
        print("-" * 34)
        for motor_id in sorted(found):
            name = id_to_name.get(motor_id, "(알 수 없음 — SO101_MOTOR_IDS에 없는 id)")
            print(f"{motor_id:<4}{name:<16}{found[motor_id]:<14}")

        expected_ids = set(SO101_MOTOR_IDS.values())
        missing = expected_ids - set(found)
        if missing:
            missing_names = [n for n, i in SO101_MOTOR_IDS.items() if i in missing]
            print(f"\n누락된 관절: {missing_names}")
        else:
            print("\n6개 관절 모두 정상 응답.")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="예: /dev/ttyACM1")
    args = parser.parse_args()
    scan_bus(args.port)
