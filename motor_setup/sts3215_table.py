# sts3215_table.py
#
# SO-101(팔로워/리더 공용)에 쓰이는 Feetech STS3215 서보의 컨트롤 테이블 중,
# "모터 ID 매핑"에 필요한 레지스터만 뽑아 정리한 것.
#
# 원본 출처: lerobot/motors/feetech/tables.py (STS_SMS_SERIES_CONTROL_TABLE)
# 전체 레지스터 문서: http://doc.feetech.cn/#/prodinfodownload?srcType=FT-SMS-STS-emanual-229f4476422d4059abfb1cb0

# 레지스터명 -> (주소, 바이트 길이)
# EPROM 영역(비휘발성, Lock=0일 때만 쓰기 가능)
MODEL_NUMBER = (3, 2)       # read-only, 모델 식별용
ID = (5, 1)
BAUD_RATE = (6, 1)
# SRAM 영역(전원 끄면 초기화)
TORQUE_ENABLE = (40, 1)
LOCK = (55, 1)               # 1=EPROM 쓰기 잠금, 0=쓰기 허용

# STS3215가 응답해야 하는 모델 번호 (다른 모델의 모터가 잘못 연결됐을 때 감지용)
EXPECTED_MODEL_NUMBER = 777

# 조립이 끝난 뒤 데이지체인 전체가 통신할 기본 baudrate.
# ID 매핑 마지막 단계에서 모든 모터를 이 값으로 통일시킨다.
DEFAULT_BAUDRATE = 1_000_000

# baudrate(bps) -> 레지스터에 쓸 값
BAUDRATE_TABLE = {
    1_000_000: 0,
    500_000: 1,
    250_000: 2,
    128_000: 3,
    115_200: 4,
    57_600: 5,
    38_400: 6,
    19_200: 7,
}

# 공장 출하 시 모터가 어떤 baudrate로 설정돼 있을지 몰라 하나씩 스캔해야 하므로,
# 시도할 순서. 실제 SO-101 조립체 기본값인 1,000,000을 가장 먼저 시도한다.
SCAN_BAUDRATES = [1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600, 38_400, 19_200]

# SO-101 6-DoF 팔의 관절명 -> 목표 ID (lerobot so_follower.py / so_leader.py와 동일한 매핑)
SO101_MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
