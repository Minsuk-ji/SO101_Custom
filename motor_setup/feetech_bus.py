# feetech_bus.py
#
# STS3215 서보와 통신하기 위한 최소한의 시리얼 버스 래퍼.
# lerobot의 motors_bus.py + feetech.py 로직 중 "모터 ID 매핑"에 필요한 부분만
# 뽑아서 하나로 합친 것 (calibration, sync_read/write, drive_mode 인코딩 등
# 다른 기능은 이 단계에서 필요 없으므로 제외했다).
#
# 실제 시리얼 통신은 lerobot과 동일하게 scservo_sdk(Feetech 공식 SDK 기반)를 사용한다.
# (lerobot conda 환경: `conda run -n lerobot python ...`)

import logging

import scservo_sdk as scs

from .sts3215_table import (
    BAUD_RATE,
    BAUDRATE_TABLE,
    DEFAULT_BAUDRATE,
    EXPECTED_MODEL_NUMBER,
    ID,
    LOCK,
    MODEL_NUMBER,
    SCAN_BAUDRATES,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 0  # STS3215는 Feetech 프로토콜 0을 사용


class FeetechBus:
    """포트 하나에 연결된 Feetech 모터와 통신하기 위한 얇은 래퍼."""

    def __init__(self, port: str):
        self.port = port
        self.port_handler = scs.PortHandler(port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION)

    def connect(self) -> None:
        if not self.port_handler.openPort():
            raise OSError(f"포트 '{self.port}'를 열 수 없습니다. 포트 경로/권한을 확인하세요.")

    def disconnect(self) -> None:
        self.port_handler.closePort()

    def set_baudrate(self, baudrate: int) -> None:
        if self.port_handler.getBaudRate() != baudrate:
            self.port_handler.setBaudRate(baudrate)

    # ── 저수준 read/write ──────────────────────────────────────────
    def read(self, address: int, length: int, motor_id: int) -> int:
        read_fn = {1: self.packet_handler.read1ByteTxRx, 2: self.packet_handler.read2ByteTxRx}[length]
        value, comm, _error = read_fn(self.port_handler, motor_id, address)
        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(self.packet_handler.getTxRxResult(comm))
        return value

    def write(self, address: int, length: int, motor_id: int, value: int) -> None:
        write_fn = {1: self.packet_handler.write1ByteTxRx, 2: self.packet_handler.write2ByteTxRx}[length]
        comm, _error = write_fn(self.port_handler, motor_id, address, value)
        if comm != scs.COMM_SUCCESS:
            raise ConnectionError(self.packet_handler.getTxRxResult(comm))

    # ── 모터 탐색 ──────────────────────────────────────────────────
    def ping(self, motor_id: int) -> int | None:
        """해당 ID로 응답하는 모터가 있으면 model number를, 없으면 None을 반환."""
        model_number, comm, _error = self.packet_handler.ping(self.port_handler, motor_id)
        return model_number if comm == scs.COMM_SUCCESS else None

    def find_connected_motor(self) -> tuple[int, int]:
        """
        버스에 모터를 딱 하나만 연결한 상태에서 호출한다.
        baudrate를 하나씩 바꿔가며 1~252번 ID를 순회 ping해서,
        응답하는 (baudrate, 현재 id)를 찾아 반환한다.

        어떤 baudrate에서도 응답이 없으면 RuntimeError
        (배선/전원 문제 — baudrate 문제는 이미 전부 시도했으므로 배제된다).
        """
        for baudrate in SCAN_BAUDRATES:
            self.set_baudrate(baudrate)
            for motor_id in range(1, 253):
                model_number = self.ping(motor_id)
                if model_number is None:
                    continue
                if model_number != EXPECTED_MODEL_NUMBER:
                    raise RuntimeError(
                        f"ID={motor_id}에서 모터를 찾았지만 model_number={model_number}로, "
                        f"STS3215({EXPECTED_MODEL_NUMBER}) 가 아닙니다. 다른 모델이 섞여 연결된 것 같습니다."
                    )
                return baudrate, motor_id

        raise RuntimeError("어떤 baudrate에서도 모터가 응답하지 않았습니다. 케이블/전원 연결을 확인하세요.")

    # ── ID 재설정 ──────────────────────────────────────────────────
    def set_motor_id(self, current_baudrate: int, current_id: int, target_id: int) -> None:
        """
        EPROM(Lock=0)을 풀고 ID 레지스터에 target_id를 쓴 뒤, baudrate까지 기본값으로
        통일하고 다시 잠근다.
        (Torque가 켜져 있으면 EPROM 쓰기가 거부되므로 먼저 Torque_Enable=0으로 끈다.)
        """
        self.set_baudrate(current_baudrate)

        torque_addr, torque_len = (40, 1)  # Torque_Enable
        self.write(torque_addr, torque_len, current_id, 0)

        lock_addr, lock_len = LOCK
        self.write(lock_addr, lock_len, current_id, 0)  # EPROM 잠금 해제

        id_addr, id_len = ID
        # ID 레지스터 자체를 바꾸는 순간 이 모터는 더 이상 current_id로 응답하지 않게 되므로,
        # 이 write가 이 함수에서 current_id를 사용하는 마지막 호출이다.
        self.write(id_addr, id_len, current_id, target_id)

        # 데이지체인 전체가 같은 속도로 통신해야 하므로, ID와 함께 baudrate도
        # 기본값(DEFAULT_BAUDRATE)으로 맞춘다. 이후 통신은 새로 부여된 target_id로 한다.
        baud_addr, baud_len = BAUD_RATE
        self.write(baud_addr, baud_len, target_id, BAUDRATE_TABLE[DEFAULT_BAUDRATE])

        self.set_baudrate(DEFAULT_BAUDRATE)
        self.write(lock_addr, lock_len, target_id, 1)  # EPROM 다시 잠금

    def read_model_number(self, motor_id: int) -> int:
        addr, length = MODEL_NUMBER
        return self.read(addr, length, motor_id)
