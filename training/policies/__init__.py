# 이 패키지를 import하면 각 정책 모듈이 실행되면서 @register_policy로
# training.registry.POLICY_REGISTRY에 자기 자신을 등록한다.
# 새 정책을 추가하려면: 이 폴더에 파일 추가 + 아래에 import 한 줄만 더하면 된다.

from . import act, mlp_bc  # noqa: F401
