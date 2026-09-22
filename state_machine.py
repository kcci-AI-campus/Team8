"""
공유 상태 기계 — 학습 모델 버전과 규칙 기반 베이스라인이 동일하게 사용합니다.
프레임 수가 아니라 '초' 단위로 판정하므로 파이의 실제 FPS가 얼마든 동작이 같습니다.
"""


class FallStateMachine:
    def __init__(self, confirm_sec: float = 1.0, miss_tolerance_sec: float = 0.3):
        self.state = "standing"
        self.since = None
        self.confirm_sec = confirm_sec
        self.miss_tolerance_sec = miss_tolerance_sec   # 이 시간 안의 미검출은 노이즈로 간주
        self.last_fall_seen = None

    def update(self, frame_label: str, t: float) -> bool:
        if frame_label == "fall":
            self.last_fall_seen = t

        if self.state == "standing":
            if frame_label == "fall":
                self.state, self.since = "falling_candidate", t
        elif self.state == "falling_candidate":
            if self.last_fall_seen is not None and t - self.last_fall_seen > self.miss_tolerance_sec:
                self.state = "standing"                 # 진짜로 fall이 끊긴 경우만 취소
            elif t - self.since >= self.confirm_sec:
                self.state = "fallen"
                return True
        elif self.state == "fallen":
            if self.last_fall_seen is not None and t - self.last_fall_seen > self.miss_tolerance_sec:
                self.state = "standing"
        return False