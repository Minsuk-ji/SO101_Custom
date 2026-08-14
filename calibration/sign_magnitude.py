# sign_magnitude.py
#
# STS3215는 부호 있는 값(Homing_Offset, Present_Position)을 2의 보수가 아니라
# sign-magnitude 방식으로 인코딩한다: 특정 비트 하나를 "부호 비트"로 예약하고,
# 나머지 하위 비트를 크기(절댓값)로 쓴다.
#
# 예) Homing_Offset(16bit 레지스터, 부호비트=bit11, 즉 하위 11비트가 크기):
#   -1082  ->  부호비트=1, 크기=1082  ->  (1 << 11) | 1082 == 3130
#
# 원본 출처: lerobot/motors/encoding_utils.py


def encode_sign_magnitude(value: int, sign_bit_index: int) -> int:
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(f"magnitude {magnitude} exceeds max {max_magnitude} for sign_bit={sign_bit_index}")

    sign_bit = 1 if value < 0 else 0
    return (sign_bit << sign_bit_index) | magnitude


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int) -> int:
    sign_bit = (encoded_value >> sign_bit_index) & 1
    magnitude = encoded_value & ((1 << sign_bit_index) - 1)
    return -magnitude if sign_bit else magnitude
