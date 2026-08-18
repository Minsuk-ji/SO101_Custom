# registry.py
#
# 이름(문자열) -> 정책 클래스 매핑. lerobot은 draccus.ChoiceRegistry로 config
# 클래스를 등록하고 이름 규칙으로 모델 클래스를 lazy-import하는데, 여기서는 그보다
# 단순한 dict 기반 레지스트리로 같은 효과(정책 종류를 train.py 수정 없이 추가)를 낸다.

from .policy_base import Policy

POLICY_REGISTRY: dict[str, type[Policy]] = {}


def register_policy(name: str):
    def decorator(cls: type[Policy]) -> type[Policy]:
        cls.name = name
        POLICY_REGISTRY[name] = cls
        return cls

    return decorator


def get_policy_class(name: str) -> type[Policy]:
    if name not in POLICY_REGISTRY:
        raise KeyError(f"알 수 없는 정책 '{name}'. 사용 가능: {list(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name]
