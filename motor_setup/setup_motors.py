# setup_motors.py
#
# SO-101 6개 관절(sts3215) 모터에 순서대로 고유 ID를 부여하는 스크립트.
# lerobot의 SOLeader.setup_motors() / SOFollower.setup_motors()와 동일한 절차를
# 직접 구현한 것 — 컨트롤러 보드에 모터를 한 번에 하나씩만 연결해서 ID를 굽는다.
#
# 실행 (lerobot conda 환경에 scservo_sdk가 설치되어 있음):
#   conda run -n lerobot python -m motor_setup.setup_motors --port /dev/ttyACM0

import argparse

from .feetech_bus import FeetechBus
from .sts3215_table import SO101_MOTOR_IDS


def setup_motors(port: str) -> None:
    bus = FeetechBus(port)
    bus.connect()

    try:
        # 그리퍼 -> 손목 -> ... -> 어깨 순으로 진행 (조립 시 케이블을 연결하는 순서와 반대 방향).
        for name, target_id in reversed(list(SO101_MOTOR_IDS.items())):
            input(f"\n컨트롤러 보드에 '{name}' 모터 하나만 연결한 뒤 Enter를 누르세요...")

            baudrate, current_id = bus.find_connected_motor()
            print(f"  -> 감지됨: id={current_id}, baudrate={baudrate}")

            if current_id == target_id:
                print(f"  -> 이미 목표 id({target_id})와 동일합니다. baudrate만 통일합니다.")

            bus.set_motor_id(baudrate, current_id, target_id)

            # 재확인: ID/baudrate 변경 후 새 설정으로 실제로 응답하는지 ping으로 검증
            if bus.ping(target_id) is None:
                raise RuntimeError(f"'{name}' ID 변경 후 재확인 ping이 실패했습니다.")

            print(f"  -> '{name}' 모터 id를 {target_id}로 설정 완료.")
    finally:
        bus.disconnect()

    print("\n모든 모터 ID 매핑 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="예: /dev/ttyACM0")
    args = parser.parse_args()
    setup_motors(args.port)
