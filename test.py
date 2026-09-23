"""시간 기반 낙상 판정 + 판정 화면을 포함한 전후 자동 녹화.
녹화물은 clips 폴더에 clip_날짜_시각 이름으로 저장됩니다
  clip_20260923_102418.avi           낙상 사건(전 3초 + 후 5초) 판정 화면
  clip_20260923_102418_raw.avi       같은 구간의 순수 캠 영상
  clip_..._manual.avi / _preview.avi 수동 녹화 / 전체 세션 녹화 (각각 _raw 동반)
녹화되는 모든 영상은 두 벌로 저장됩니다 — <이름>.avi(판정 화면)와 <이름>_raw.avi(순수 캠 영상).
_raw 쪽은 다른 판정 방식으로 같은 장면을 다시 실험할 때 그대로 넣어 쓰면 됩니다. (--no-raw로 끔)
실행: python test_ver8.py --source 0
영상: python test_ver8.py --source sample.avi --headless --save-preview
모델 경로는 --model / --pose-model로 지정 가능. 상세 내용: CHANGES_KO.md
"""
import time
import os
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from collections import deque

# 추론 런타임은 main()에서 로드: 상태/녹화 테스트에는 모델이 필요 없음.
tflite = None

# ---- 설정: 시간 관련 추가 설정은 TemporalFallDetector 바로 위에 있음 ----
MODEL_PATH = "best_Hfall_int8.tflite"
POSE_MODEL_PATH = "yolo11n-pose_int8.tflite"
CONF_TH = 0.45
IOU_TH = 0.45
EMERGENCY_SEC = 5.0  # 낙상 확정 이후 관측된 넘어진 상태 N초
CONFIRM_SEC = 1.0
INFO_PAD_HEIGHT = 76
DEBUG = False
WINDOW_NAME = "fall detection"
REC_FOURCC = "MJPG"
SAVE_RAW = False  # 판정 화면과 함께 순수 캠 영상(_raw)도 저장 (--no-raw로 끔)
FREEZE_ON_ALARM = True  # 표시만 3초 고정; 추론/녹화는 계속
FREEZE_DURATION_SEC = 3.0
POSE_IMG_SIZE = 192  # 좌표 복원용; 실제 입력 크기는 모델 메타데이터에서 읽음
POSE_CONF_TH = 0.5
KPT_CONF_TH = 0.4
LYING_ANGLE_TH_DEG = 55
LEG_ANGLE_TH_DEG = 60
KPT_ASPECT_TH = 1.0
LYING_VOTE_MIN = 2
POSE_PAD_RATIO = 0.15

CLASS_NAMES = {0: "fall", 1: "non-fall"}
FALL_ID = [k for k, v in CLASS_NAMES.items() if v == "fall"][0]
BOX_COLOR = {
    "fall": (0, 0, 255),       # 빨강 (낙상)
    "non-fall": (0, 255, 0),   # 초록 (정상)
    "static": (128, 128, 128)  # 회색 (가만히 있는 물체/배경)
}
# [추가] 판정 상태별 색 — 박스·라벨·상단 글자·하단 막대가 모두 이 색을 쓴다.
STATE_COLOR = {
    'emergency': (0, 0, 255),
    'unknown': (160, 160, 160),          # 관측 누락/판단 보류
    'recovering': (255, 180, 0),         # 회복 자세 확인 중
    'standing': (0, 255, 0),            # 초록 (정상)
    'falling_candidate': (0, 165, 255),  # 주황 (낙상 의심)
    'fallen': (0, 0, 255),               # 빨강 (낙상 확정)
}
SKELETON_COLOR = (255, 255, 0)           # 청록 — 자세 정보라 판정 색과 구분

# [추가: 포즈] COCO 17 키포인트 인덱스 & 스켈레톤 연결선 (시각화/각도 계산용)
KPT = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # 어깨-팔
    (5, 11), (6, 12), (11, 12),                # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),    # 다리
    (0, 5), (0, 6),                            # 코-어깨
]


# [추가] 하단 막대 UI + 버튼 ------------------------------------------------
BUTTONS = {}      # 이름 -> (x1, y1, x2, y2) — draw_bar에서 매 프레임 갱신
_click = None     # 마지막 마우스 클릭 좌표


def on_mouse(event, x, y, flags, param):
    global _click
    if event == cv2.EVENT_LBUTTONDOWN:
        _click = (x, y)


def clicked(name, pos):
    if pos is None or name not in BUTTONS:
        return False
    x1, y1, x2, y2 = BUTTONS[name]
    return x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2


def read_actions(key):
    """키 입력과 마우스 클릭을 합쳐서 (녹화 토글, 종료) 반환."""
    global _click
    pos, _click = _click, None
    toggle = clicked("rec", pos) or key in (ord("r"), ord(" "))
    quit_now = clicked("quit", pos) or key == ord("q")
    return toggle, quit_now


def draw_bar(canvas, h, w, state_line, rec_line, state_color, recording):
    """하단 검은 막대: 왼쪽에 버튼 2개, 오른쪽에 상태 두 줄."""
    cv2.rectangle(canvas, (0, h), (w, h + INFO_PAD_HEIGHT), (0, 0, 0), -1)

    by1, by2 = h + 18, h + 58
    BUTTONS["rec"] = (10, by1, 175, by2)
    BUTTONS["quit"] = (185, by1, 265, by2)

    rec_color = (0, 0, 220) if not recording else (0, 140, 220)
    cv2.rectangle(canvas, (10, by1), (175, by2), rec_color, -1)
    cv2.putText(canvas, "STOP & SAVE" if recording else "START REC", (22, by2 - 13),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)

    cv2.rectangle(canvas, (185, by1), (265, by2), (70, 70, 70), -1)
    cv2.putText(canvas, "QUIT", (203, by2 - 13), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)

    cv2.putText(canvas, state_line, (280, h + 32), cv2.FONT_HERSHEY_PLAIN, 1.3, state_color, 2)
    cv2.putText(canvas, rec_line, (280, h + 58), cv2.FONT_HERSHEY_PLAIN, 1.3,
                (0, 0, 255) if recording else (200, 200, 200), 2)

    # 녹화 중 표시: 영상 오른쪽 위 빨간 점 (0.5초 간격 깜빡임)
    if recording and int(time.time() * 2) % 2 == 0:
        cv2.circle(canvas, (w - 25, 25), 10, (0, 0, 255), -1)


# [추가: 포즈] ---------------------------------------------------------------
def load_pose_interpreter():
    interp = tflite.Interpreter(model_path=POSE_MODEL_PATH)
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    print("[pose] input :", in_d)
    print("[pose] output:", out_d)
    return interp, in_d, out_d


def _estimate_pose_single(interp, in_d, out_d, frame, x1, y1, x2, y2):
    """
    프레임에서 (x1,y1,x2,y2) 박스를 여유 있게 잘라 포즈 모델에 넣고,
    17개 키포인트를 (원본 프레임 좌표 x, y, confidence) 배열로 반환.
    크롭 안에서 사람이 확인 안 되거나 신뢰도가 낮으면 None.
    """
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * POSE_PAD_RATIO), int(bh * POSE_PAD_RATIO)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (POSE_IMG_SIZE, POSE_IMG_SIZE))

    raw = infer_tensor(interp, in_d, out_d, crop)

    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] == 56:
        raw = raw.transpose()   # (N, 56) = [cx,cy,w,h, person_conf, (kx,ky,kconf)*17]

    if raw.shape[0] == 0:
        return None

    if raw.shape[1] != 56:
        raise ValueError(f"Expected pose output (N,56), got {raw.shape}")
    person_conf = raw[:, 4]
    best_i = int(np.argmax(person_conf))
    if person_conf[best_i] < POSE_CONF_TH:
        return None

    kpts = raw[best_i, 5:].reshape(17, 3)
    scale_x = crop_w / POSE_IMG_SIZE
    scale_y = crop_h / POSE_IMG_SIZE

    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 0] = kpts[:, 0] * POSE_IMG_SIZE * scale_x + cx1  # 원본 프레임 x
    keypoints[:, 1] = kpts[:, 1] * POSE_IMG_SIZE * scale_y + cy1  # 원본 프레임 y
    keypoints[:, 2] = kpts[:, 2]

    return keypoints


def estimate_pose(interp, in_d, out_d, frame, x1, y1, x2, y2):
    """옆으로 누운 자세에서 실패하면 사람 영역을 90도 회전해 재추론.
    관절은 원본 좌표로 복원하므로 각도/누움 판정은 원래 화면 기준이다.
    """
    keypoints = _estimate_pose_single(interp, in_d, out_d, frame, x1, y1, x2, y2)
    if keypoints is not None:
        return keypoints
    h, w = frame.shape[:2]
    px, py = int((x2-x1)*POSE_PAD_RATIO), int((y2-y1)*POSE_PAD_RATIO)
    ax, ay = max(0,x1-px), max(0,y1-py)
    bx, by = min(w,x2+px), min(h,y2+py)
    crop = frame[ay:by, ax:bx]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    choices = []
    for direction in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotated = cv2.rotate(crop, direction)
        k = _estimate_pose_single(interp, in_d, out_d, rotated, 0, 0, ch, cw)
        if k is None or int((k[:, 2] >= KPT_CONF_TH).sum()) < 5:
            continue
        u, v = k[:, 0].copy(), k[:, 1].copy()
        if direction == cv2.ROTATE_90_CLOCKWISE:
            k[:, 0], k[:, 1] = v + ax, ch - 1 - u + ay
        else:
            k[:, 0], k[:, 1] = cw - 1 - v + ax, u + ay
        choices.append(k)
    return max(choices, key=lambda k: float(k[:, 2].sum())) if choices else None


def pose_search_box(keypoints, person_box, frame_shape):
    """마지막 관절 영역을 확장. 가구까지 포함한 검출 박스보다 사람에 집중."""
    visible = keypoints[keypoints[:, 2] >= KPT_CONF_TH]
    if len(visible) < 5:
        return None
    h, w = frame_shape[:2]
    scale = max(person_box[3] - person_box[1], 1)
    low, high = visible[:, :2].min(axis=0), visible[:, :2].max(axis=0)
    return (max(0, int(low[0] - scale*.35)), max(0, int(low[1] - scale*.25)),
            min(w, int(high[0] + scale*.35)), min(h, int(high[1] + scale*.5)))


def _kpt_center(keypoints, name_a, name_b):
    """두 키포인트 중 신뢰도 충분한 것들의 중심. 둘 다 안 보이면 None."""
    pts = [keypoints[KPT[name_a]], keypoints[KPT[name_b]]]
    vis = [p for p in pts if p[2] >= KPT_CONF_TH]
    if not vis:
        return None
    return (float(np.mean([p[0] for p in vis])), float(np.mean([p[1] for p in vis])))


def _angle_from_vertical(p_top, p_bottom):
    dx = p_bottom[0] - p_top[0]
    dy = p_bottom[1] - p_top[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def is_lying_down(keypoints):
    """
    [수정] 몸통 각도 하나 대신 3개 지표로 투표해서 "바닥에 누운/쓰러진 자세"를 판단.
      1) 몸통 각도  : 어깨중심→엉덩이중심 축이 수직선과 벌어진 각도
      2) 다리 각도  : 엉덩이중심→발목중심 축이 수직선과 벌어진 각도 (다리를 바닥에 뻗으면 90도 근처)
      3) 가로세로비 : 보이는 키포인트 전체를 감싸는 박스의 가로/세로
    계산 가능한 지표가 2개 미만이면 None(판단 보류).
    반드시 Python bool을 반환한다 — np.bool_를 반환하면 `is True` 비교가 항상 실패함.
    """
    votes, checks = 0, 0
    info = []

    shoulder_c = _kpt_center(keypoints, "l_shoulder", "r_shoulder")
    hip_c = _kpt_center(keypoints, "l_hip", "r_hip")
    ankle_c = _kpt_center(keypoints, "l_ankle", "r_ankle")

    if shoulder_c and hip_c:
        torso = _angle_from_vertical(shoulder_c, hip_c)
        checks += 1
        votes += int(torso > LYING_ANGLE_TH_DEG)
        info.append(f"torso={torso:.0f}°")

    if hip_c and ankle_c:
        leg = _angle_from_vertical(hip_c, ankle_c)
        checks += 1
        votes += int(leg > LEG_ANGLE_TH_DEG)
        info.append(f"leg={leg:.0f}°")

    vis = keypoints[keypoints[:, 2] >= KPT_CONF_TH]
    if len(vis) >= 5:
        w = float(np.ptp(vis[:, 0]))
        h = float(np.ptp(vis[:, 1]))
        aspect = w / (h + 1e-6)
        checks += 1
        votes += int(aspect > KPT_ASPECT_TH)
        info.append(f"aspect={aspect:.2f}")

    if DEBUG:
        print(f"[DEBUG] pose 지표: {' '.join(info) or '(계산 불가)'}  votes={votes}/{checks}")

    if checks < 2:
        return None
    return bool(votes >= LYING_VOTE_MIN)


def draw_skeleton(frame, keypoints, color=SKELETON_COLOR):
    for x, y, c in keypoints:
        if c >= KPT_CONF_TH:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    for a, b in SKELETON_EDGES:
        if keypoints[a][2] >= KPT_CONF_TH and keypoints[b][2] >= KPT_CONF_TH:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pt1, pt2, color, 2)
# -----------------------------------------------------------------------------


# ---- 추가: 시간 기반 판정 / 자동 사건 녹화 -------------------------------
PRE_EVENT_SEC = 3.0
POST_EVENT_SEC = 5.0
HISTORY_SEC = 1.5
DROP_RATIO = 0.18       # 이전 사람 박스 높이 대비 중심 하강량
DROP_SPEED = 0.20       # 초당 이전 박스 높이 대비 하강량
ANGLE_CHANGE = 20.0
RECOVERY_SEC = 1.0
MAX_GAP_SEC = 0.75
EVENT_FPS = 10.0
BUFFER_MAX_FRAMES = 90  # JPEG 압축 프레임으로 메모리 사용 제한
# [추가] 낙상이 확정되지 않던 문제 대응 설정
TARGET_IOU = 0.05          # 대상 추적 매칭 임계값 (넘어지면 박스 모양이 확 바뀌어 IoU가 끊김)
REACQUIRE_DIST_RATIO = 1.2  # IoU가 끊겨도 중심이 이 비율(직전 박스 높이 대비) 안이면 같은 사람으로 이어붙임
LYING_CONFIRM_SEC = 1.5    # 하강 순간을 놓쳤어도 '누움'이 이만큼 이어지면 낙상 확정
LYING_GRACE_SEC = 0.6      # 누움 사이에 '안 누움/관측 실패'가 이 시간 이내면 끊긴 것으로 보지 않음
                           # (int8 포즈가 프레임마다 흔들리고, 바닥에 있으면 가려져 키포인트가 자주 빠짐)
CLASS_MERGE_IOU = 0.4      # 같은 사람에 fall/non-fall 박스가 겹쳐 나올 때 같은 사람으로 묶는 기준
TARGET_STUCK_SEC = 2.0     # 대상 박스에서 포즈가 이 시간 이상 안 잡히면 대상 포기 (정지 사물 고착 방지)
TARGET_BLOCK_SEC = 5.0     # 포기한 박스 영역은 이 시간 동안 새 대상으로 뽑지 않음

# [수정] 빠른 낙상과 회복을 위한 별도 확인 조건.
RECOVERY_ANGLE_DEG = 35.0
DOWN_ANGLE_DEG = 50.0
DOWN_ASPECT_RATIO = 1.0
FAST_CONFIRM_SEC = 0.4  # 하강+자세변화가 있을 때만 사용, 관측 누락 시간 제외
SEATED_HIP_DROP_RATIO = 0.10
SEATED_SHOULDER_DROP_RATIO = 0.18  # 골반/어깨 동반 하강으로 허리 숙이기와 구별
CANDIDATE_HOLD_SEC = 3.0
POSE_FALLBACK_SEC = 2.0  # 최근 대상 영역에서 짧게 포즈 재확인


def clip_path(directory, suffix='', ext='.avi'):
    """clips 폴더에 clip_날짜_시각[_구분].확장자 형태의 겹치지 않는 경로를 만든다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    path = directory / f'clip_{stamp}{suffix}{ext}'
    index = 2
    while path.exists():
        path = directory / f'clip_{stamp}_{index}{suffix}{ext}'
        index += 1
    return path


def infer_tensor(interp, inputs, outputs, bgr):
    """모델의 NHWC/NCHW 형태, 양자화 입출력에 맞춘 전처리/후처리.
    기존 모델과 같이 0..1 RGB 및 직접 resize를 사용한다.
    """
    detail = inputs[0]
    shape = list(detail['shape'])
    if len(shape) != 4:
        raise ValueError(f'Unsupported input shape: {shape}')
    if shape[-1] == 3:
        height, width, nchw = shape[1], shape[2], False
    elif shape[1] == 3:
        height, width, nchw = shape[2], shape[3], True
    else:
        raise ValueError(f'Expected RGB input: {shape}')
    data = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (width, height))
    data = data.astype(np.float32) / 255.0
    dtype = detail['dtype']
    if np.issubdtype(dtype, np.integer):
        scale, zero = detail['quantization']
        if scale <= 0:
            raise ValueError('Invalid input quantization scale')
        limits = np.iinfo(dtype)
        data = np.clip(np.rint(data / scale + zero), limits.min, limits.max)
    data = data.astype(dtype)[None]
    if nchw:
        data = data.transpose(0, 3, 1, 2)
    interp.set_tensor(detail['index'], data)
    interp.invoke()
    out = interp.get_tensor(outputs[0]['index'])
    if np.issubdtype(out.dtype, np.integer):
        scale, zero = outputs[0]['quantization']
        if scale <= 0:
            raise ValueError('Invalid output quantization scale')
        out = (out.astype(np.float32) - zero) * scale
    return out


class TemporalFallDetector:
    """관측 누락/회복/낙상 확정을 분리. 프레임 수 대신 영상 시각 사용."""
    def __init__(self, emergency_sec=EMERGENCY_SEC):
        if not np.isfinite(emergency_sec) or emergency_sec <= 0:
            raise ValueError('emergency_sec must be finite and positive')
        self.emergency_sec = float(emergency_sec)
        self.fallen_duration = 0.0
        self.emergency_active = False
        self.emergency_now = False
        self._last_fallen_t = None
        self._last_fallen_observed = None
        self.state = 'unknown'
        self.history = deque()
        self.last_seen = None
        self.lying_since = None
        self.lying_run_since = None
        self.last_lying_t = None
        self.recover_since = None
        self.candidate_since = None
        self.last_descent_t = None
        self.evidence_sec = 0.0
        self.previous_support_t = None
        self.last_support_t = None
        self.confirmed = False
        self.reason = 'waiting for person/pose'

    def _clear_evidence(self):
        self.lying_run_since = self.last_lying_t = None
        self.previous_support_t = self.last_support_t = None
        self.evidence_sec = 0.0

    def update(self, t, obs):
        """N초는 낙상 확정 이후부터 계산. 관측 실패 시간은 합산하지 않는다.
        긴 누락/회복 자세는 확정 전 타이머 초기화. Emergency는 회복 확정까지 유지.
        """
        self.emergency_now = False
        alarm = self._update_fall(t, obs)
        if not self.confirmed:
            self.fallen_duration = 0.0
            self._last_fallen_t = self._last_fallen_observed = None
            self.emergency_active = False
        elif obs is None:
            self._last_fallen_t = None
            if (self._last_fallen_observed is not None
                    and t - self._last_fallen_observed > MAX_GAP_SEC):
                self.fallen_duration = 0.0
        elif self.state == 'recovering':
            self.fallen_duration = 0.0
            self._last_fallen_t = self._last_fallen_observed = None
        elif self.state == 'fallen':
            if (self._last_fallen_observed is not None
                    and t - self._last_fallen_observed > MAX_GAP_SEC):
                self.fallen_duration = 0.0
                self._last_fallen_t = None
            if self._last_fallen_t is not None:
                self.fallen_duration += t - self._last_fallen_t
            self._last_fallen_t = self._last_fallen_observed = t
            if not self.emergency_active and self.fallen_duration >= self.emergency_sec - 1e-7:
                self.emergency_active = self.emergency_now = True
        if self.emergency_active:
            self.reason = ('emergency - person/pose missing' if obs is None else
                           'emergency - checking recovery' if self.state == 'recovering' else
                           'emergency - prolonged fallen state')
            self.state = 'emergency'
        return alarm

    def _update_fall(self, t, obs):
        while self.history and t - self.history[0][0] > HISTORY_SEC:
            self.history.popleft()
        if obs is None:
            # 누락 시간을 누움 지속 시간에 더하지 않는다.
            self.previous_support_t = None
            self.recover_since = None
            self.reason = 'person/pose missing - not normal'
            if self.confirmed:
                self.state = 'fallen'
                self.reason = 'confirmed fall - pose lost, awaiting recovery'
            elif (not self.confirmed and self.candidate_since is not None
                    and t - self.candidate_since <= CANDIDATE_HOLD_SEC):
                self.state = 'falling_candidate'
            else:
                self.state = 'unknown'
            if self.last_support_t is not None and t - self.last_support_t > LYING_GRACE_SEC:
                self._clear_evidence()
            return False

        gap = None if self.last_seen is None else t - self.last_seen
        self.last_seen = t
        angle = obs['angle']
        # fall 점수가 남더라도 신뢰되는 수직 몸통 자세면 회복 가능.
        upright = (obs['lying'] is False and angle is not None
                   and angle <= RECOVERY_ANGLE_DEG)
        descent = False
        abrupt_tilt = False
        for old_t, old in self.history:
            dt = t - old_t
            drop = (obs['cy'] - old['cy']) / max(old['height'], 1)
            angle_change = (angle is not None and old['angle'] is not None
                            and angle - old['angle'] >= ANGLE_CHANGE)
            shape_change = obs['aspect'] - old['aspect'] >= 0.35
            if (angle_change and obs['fall'] and old['angle'] <= RECOVERY_ANGLE_DEG
                    and angle >= DOWN_ANGLE_DEG):
                abrupt_tilt = True
            shoulder_drop = None
            if obs.get('shoulder_y') is not None and old.get('shoulder_y') is not None:
                shoulder_drop = (obs['shoulder_y'] - old['shoulder_y']) / max(old['height'], 1)
            seated_drop = (drop >= SEATED_HIP_DROP_RATIO and shoulder_drop is not None
                           and shoulder_drop >= SEATED_SHOULDER_DROP_RATIO and angle_change)
            if (dt >= 0.08 and (drop >= DROP_RATIO or seated_drop) and drop / dt >= DROP_SPEED
                    and (angle_change or shape_change) and old['lying'] is False
                    and old['angle'] is not None and old['angle'] <= RECOVERY_ANGLE_DEG):
                descent = True
                break
        self.history.append((t, dict(obs)))
        if abrupt_tilt and not self.confirmed and self.candidate_since is None:
            # 급격한 기울기는 의심만 시작. 빠른 확정에는 실제 하강도 필요.
            self.candidate_since = t
        if descent and not self.confirmed:
            self.last_descent_t = t
            if self.candidate_since is None:
                self.candidate_since = t
            self.state = 'falling_candidate'
            self.reason = 'descent + posture change'
        recent_descent = (self.last_descent_t is not None
                          and t - self.last_descent_t <= CANDIDATE_HOLD_SEC)
        # 하강을 실제 관측한 경우에만 다리가 굽혀진 낙상 자세도 확인에 사용.
        down_pose = (obs['lying'] is True or
                     (angle is not None and angle >= DOWN_ANGLE_DEG
                      and obs.get('pose_aspect', 0) >= DOWN_ASPECT_RATIO))
        support = bool(down_pose and (obs['fall'] or recent_descent) and not upright)
        if support:
            if self.last_support_t is None or t - self.last_support_t > LYING_GRACE_SEC:
                self._clear_evidence()
            if self.lying_run_since is None:
                self.lying_run_since = t
            # 긴 누락/안 누움 구간은 경과 초에서 제외. 연속 관측 구간만 적립.
            if self.previous_support_t is not None:
                dt = t - self.previous_support_t
                if dt <= MAX_GAP_SEC:
                    self.evidence_sec += dt
            self.previous_support_t = self.last_support_t = self.last_lying_t = t
        else:
            self.previous_support_t = None
            if upright or self.last_support_t is None or t - self.last_support_t > LYING_GRACE_SEC:
                self._clear_evidence()

        if self.confirmed:
            if upright:
                if self.recover_since is None or (gap is not None and gap > MAX_GAP_SEC):
                    self.recover_since = t
                self.state = 'recovering'
                self.reason = 'upright pose - confirming recovery'
                if t - self.recover_since >= RECOVERY_SEC - 1e-7:
                    self.confirmed = False
                    self.state = 'standing'
                    self.reason = 'recovered by upright pose'
                    self.history.clear()
                    self._clear_evidence()
                    self.candidate_since = self.last_descent_t = None
                    self.recover_since = None
            else:
                self.recover_since = None
                self.state = 'fallen'
                self.reason = 'confirmed fall - awaiting recovery'
            return False

        required = FAST_CONFIRM_SEC if recent_descent else LYING_CONFIRM_SEC
        if support and self.evidence_sec >= required - 1e-7:
            self.confirmed = True
            self.state = 'fallen'
            self.recover_since = None
            self.reason = 'descent + observed down posture' if recent_descent else 'sustained lying'
            return True
        if upright:
            self.state = 'standing'
            self.candidate_since = self.last_descent_t = None
            self.reason = 'upright pose'
        elif (support or recent_descent or
              (self.candidate_since is not None and t - self.candidate_since <= CANDIDATE_HOLD_SEC)):
            self.state = 'falling_candidate'
            self.reason = 'checking down posture'
        else:
            self.state = 'unknown'
            self.candidate_since = None
            self.reason = 'posture uncertain'
        return False

class TimedVideo:
    """관측 시각을 고정 FPS 영상으로 변환. 빈 구간은 직전 프레임 유지."""
    def __init__(self, path, fps=EVENT_FPS):
        self.path, self.fps = str(path), fps
        self.writer = None
        self.previous = None
        self.start = self.last = self.next_tick = None
        self.count = 0

    def add(self, t, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*REC_FOURCC), self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f'Cannot open video writer: {self.path}')
            self.start = self.next_tick = t
        if self.previous is not None:
            while self.next_tick < t - 1e-7:
                self.writer.write(self.previous)
                self.count += 1
                self.next_tick += 1 / self.fps
        self.previous = frame.copy()
        self.last = t

    def close(self):
        if self.writer is not None:
            self.writer.write(self.previous)
            self.count += 1
            self.writer.release()
            self.writer = None
            print(f'[video] {self.path} ({self.count} frames)')


class PairedVideo:
    """같은 시간축으로 두 영상을 함께 저장.
      <name>.avi      — 판정 화면(박스·스켈레톤·상태 표시 포함)
      <name>_raw.avi  — 표시 없는 순수 캠 영상 (다른 방식으로 다시 실험할 때 사용)
    TimedVideo를 두 개 들고 있을 뿐이라 프레임 시각 처리는 기존과 동일하다.
    """
    def __init__(self, path, fps=EVENT_FPS, save_raw=None):
        path = Path(path)
        self.overlay = TimedVideo(path, fps)
        keep_raw = SAVE_RAW if save_raw is None else save_raw
        self.raw = TimedVideo(path.with_name(path.stem + '_raw' + path.suffix), fps) if keep_raw else None

    def add(self, t, canvas, original=None):
        self.overlay.add(t, canvas)
        if self.raw is not None and original is not None:
            self.raw.add(t, original)

    def close(self):
        self.overlay.close()
        if self.raw is not None:
            self.raw.close()

    # EventRecorder가 TimedVideo처럼 다룰 수 있도록 속성 위임
    @property
    def path(self):
        return self.overlay.path

    @property
    def raw_path(self):
        return self.raw.path if self.raw is not None else None

    @property
    def previous(self):
        return self.overlay.previous

    @property
    def raw_previous(self):
        return self.raw.previous if self.raw is not None else None

    @property
    def start(self):
        return self.overlay.start

    @property
    def last(self):
        return self.overlay.last


class EventRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.buffer = deque(maxlen=BUFFER_MAX_FRAMES)
        self.active = None
        self.deadline = None
        self.metadata = None
        self.completed = []
        self.last_buffer_time = None

    def update(self, t, canvas, original=None, alarm=False, reason=''):
        # 사건 이후 5초가 지난 첫 프레임은 종료 경계까지만 이전 화면으로 채움.
        if self.active is not None and t > self.deadline:
            self.active.add(self.deadline, self.active.previous, self.active.raw_previous)
            self.finish(False)
        if self.last_buffer_time is None or t - self.last_buffer_time >= 1 / EVENT_FPS - 1e-7 or alarm:
            ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError('Cannot encode pre-event frame')
            encoded_raw = None
            if SAVE_RAW and original is not None:
                ok_raw, encoded_raw = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok_raw:
                    raise RuntimeError('Cannot encode pre-event raw frame')
            self.buffer.append((t, encoded, encoded_raw))
            self.last_buffer_time = t
        # 경계 직전 프레임 하나를 유지해 가능한 경우 정확히 3초부터 시작.
        while len(self.buffer) > 1 and self.buffer[1][0] <= t - PRE_EVENT_SEC:
            self.buffer.popleft()
        if alarm:
            if self.active is None:
                self.active = PairedVideo(clip_path(self.directory))
                begin = max(t - PRE_EVENT_SEC, self.buffer[0][0])
                self.metadata = {'trigger_time': t, 'start_time': begin, 'reason': reason,
                                 'overlay': True, 'raw_path': self.active.raw_path,
                                 'requested_pre_sec': PRE_EVENT_SEC,
                                 'requested_post_sec': POST_EVENT_SEC, 'triggers': [t]}
                for stamp, jpg, jpg_raw in self.buffer:
                    self.active.add(max(stamp, begin), cv2.imdecode(jpg, cv2.IMREAD_COLOR),
                                    cv2.imdecode(jpg_raw, cv2.IMREAD_COLOR) if jpg_raw is not None else None)
            else:
                self.metadata['triggers'].append(t)
                self.active.add(t, canvas, original)
            self.deadline = t + POST_EVENT_SEC
        elif self.active is not None:
            self.active.add(t, canvas, original)
        if self.active is not None and t >= self.deadline:
            self.finish(False)

    def finish(self, interrupted=True):
        if self.active is None:
            return
        video = self.active
        self.metadata.update(end_time=video.last, interrupted=interrupted,
                             actual_pre_sec=self.metadata['trigger_time'] - video.start,
                             actual_post_sec=video.last - self.metadata['triggers'][-1])
        video.close()
        Path(video.path).with_suffix('.json').write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        self.completed.append(video.path)
        if video.raw_path:
            self.completed.append(video.raw_path)
        self.active = None


def detect_people(interp, inputs, outputs, frame):
    raw = infer_tensor(interp, inputs, outputs, frame)[0]
    if raw.shape[0] == 4 + len(CLASS_NAMES):
        raw = raw.T
    if raw.ndim != 2 or raw.shape[1] != 4 + len(CLASS_NAMES):
        raise ValueError(f'Expected YOLO detection output with 6 fields: {raw.shape}')
    scores = raw[:, 4:].max(axis=1)
    ids = raw[:, 4:].argmax(axis=1)
    raw, scores, ids = raw[scores > CONF_TH], scores[scores > CONF_TH], ids[scores > CONF_TH]
    h, w = frame.shape[:2]
    boxes = [[float((r[0]-r[2]/2)*w), float((r[1]-r[3]/2)*h),
              float(r[2]*w), float(r[3]*h)] for r in raw]
    # 두 클래스 모두 사람: 클래스 간 중복 박스도 제거.
    keep = cv2.dnn.NMSBoxes(boxes, scores.tolist(), CONF_TH, IOU_TH)
    xyxy = np.array([[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in boxes], dtype=np.float32) \
        if boxes else np.zeros((0, 4), np.float32)
    result = []
    for i in np.asarray(keep).reshape(-1):
        x, y, bw, bh = boxes[i]
        box = (max(0, int(x)), max(0, int(y)), min(w, int(x+bw)), min(h, int(y+bh)))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        # [수정] 클래스 무관 NMS는 한 사람에 fall/non-fall 박스가 겹쳐 나올 때 점수 높은 쪽만 남긴다.
        # 그 바람에 fall 근거가 통째로 사라져 낙상이 확정되지 않았다.
        # 이 박스와 겹치는 후보들 중 fall 클래스의 최고 점수를 따로 구해 함께 넘긴다.
        fall_score = 0.0
        if len(xyxy):
            ax1, ay1, ax2, ay2 = xyxy[i]
            ix1 = np.maximum(ax1, xyxy[:, 0]); iy1 = np.maximum(ay1, xyxy[:, 1])
            ix2 = np.minimum(ax2, xyxy[:, 2]); iy2 = np.minimum(ay2, xyxy[:, 3])
            inter = np.clip(ix2-ix1, 0, None) * np.clip(iy2-iy1, 0, None)
            area_a = max((ax2-ax1) * (ay2-ay1), 1e-6)
            area_b = (xyxy[:, 2]-xyxy[:, 0]) * (xyxy[:, 3]-xyxy[:, 1])
            overlap = inter / np.maximum(area_a + area_b - inter, 1e-6)
            same = (overlap >= CLASS_MERGE_IOU) & (ids == FALL_ID)
            if same.any():
                fall_score = float(scores[same].max())
        result.append((box, float(scores[i]), int(ids[i]), fall_score))
    return result


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def box_center_dist(a, b):
    (ax, ay), (bx, by) = box_center(a), box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def is_blocked(box, blocked, t):
    """포즈가 잡히지 않아 포기한 영역인지 (정지 사물에 대상이 고착되는 것 방지)."""
    return any(until > t and box_iou(box, region) >= 0.5 for region, until in blocked)


def box_iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / max(1, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-area)


def draw_emergency_status(canvas, sm, t):
    """기존 영상/UI에 긴급 배너와 테두리. 녹화에도 동일하게 포함."""
    h, w = canvas.shape[:2]
    if sm.emergency_active:
        red = (0, 0, 230) if int(t * 2) % 2 == 0 else (0, 0, 120)
        cv2.rectangle(canvas, (0, 0), (w-1, h-1), red, 8)
        cv2.rectangle(canvas, (0, 0), (w-1, 72), red, -1)
        cv2.putText(canvas, 'EMERGENCY - HELP REQUIRED', (12, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        detail = ('RECOVERY CHECK' if 'recovery' in sm.reason else
                  'POSE LOST - ALERT REMAINS' if 'missing' in sm.reason else
                  f'FALL PERSISTED >= {sm.emergency_sec:g}s')
        cv2.putText(canvas, detail, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    elif sm.confirmed:
        cv2.putText(canvas, f'Emergency timer: {sm.fallen_duration:.1f}/{sm.emergency_sec:g}s',
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 200, 255), 2)


def main():
    global tflite, POSE_MODEL_PATH, SAVE_RAW, CLASS_NAMES, FALL_ID
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument('--source', default='0', help='camera index or video path')
    parser.add_argument('--model', default=str(base / MODEL_PATH))
    parser.add_argument('--pose-model', default=str(base / POSE_MODEL_PATH))
    parser.add_argument('--output-dir', default=str(base / 'clips'))
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--save-preview', action='store_true', help='save full annotated video')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--emergency-sec', type=float, default=EMERGENCY_SEC, help='seconds of observed fallen state before Emergency')
    parser.add_argument('--no-raw', action='store_true',
                        help='순수 캠 영상(_raw) 저장을 끔 (기본은 판정 화면과 함께 저장)')
    args = parser.parse_args()
    if not np.isfinite(args.emergency_sec) or args.emergency_sec <= 0:
        parser.error('--emergency-sec must be finite and positive')
    SAVE_RAW = not args.no_raw
    for path in (args.model, args.pose_model):
        if not Path(path).is_file():
            parser.error(f'Model file missing: {path}')
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            import tensorflow as tf
            tflite = tf.lite
    # 이름이 같은 다른 모델을 실수로 사용하는 경우 안내 (클래스 수가 같으면 경고만 하고 계속 진행).
    import zipfile
    if zipfile.is_zipfile(args.model):
        with zipfile.ZipFile(args.model) as archive:
            if 'metadata.json' in archive.namelist():
                metadata = json.loads(archive.read('metadata.json'))
                names = metadata.get('names')
                expected = {str(k): v for k, v in CLASS_NAMES.items()}
                if names is not None:
                    actual = {int(k): v for k, v in names.items()}
                    normalized = {k: v.lower().replace('_', '-').strip() for k, v in actual.items()}
                    fall_ids = [k for k, v in normalized.items() if v in ('fall', 'fall-detected')]
                    others = [v for k, v in normalized.items() if k not in fall_ids]
                    if len(actual) != 2 or len(fall_ids) != 1 or any(v not in ('person', 'non-fall') for v in others):
                        parser.error(f'Unsupported model classes: {actual}')
                    CLASS_NAMES, FALL_ID = actual, fall_ids[0]
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    POSE_MODEL_PATH = args.pose_model
    pose, pose_inputs, pose_outputs = load_pose_interpreter()
    print('[fall-model]', inputs, outputs)
    is_camera = args.source.isdecimal()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    # cap = cv2.VideoCapture('./clips/test.avi')
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open source: {args.source}')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not is_camera and (not np.isfinite(source_fps) or source_fps <= 0):
        cap.release()
        raise RuntimeError('Video FPS missing; cannot construct reliable timestamps')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = EventRecorder(out_dir)
    sm = TemporalFallDetector(emergency_sec=args.emergency_sec)
    preview = PairedVideo(clip_path(out_dir, '_preview')) if args.save_preview else None
    manual = None
    log = (out_dir / 'predictions.jsonl').open('w', encoding='utf-8')
    frozen, freeze_until = None, 0
    target, target_seen = None, None
    last_valid_box = None
    last_pose_roi = None
    last_valid_pose_t = None
    pose_fail_since = None   # [추가] 대상 박스에서 포즈가 안 잡히기 시작한 시각
    blocked = []             # [추가] 포즈가 안 잡혀 포기한 영역 [(box, 해제 시각)]
    frames = alarms = emergencies = 0
    # [추가] 낙상이 확정되지 않을 때 원인을 바로 볼 수 있는 집계
    diag = dict(frames_with_detection=0, frames_with_pose=0, frames_fall_evidence=0,
                frames_lying=0, frames_fall_and_lying=0, longest_lying_run_sec=0.0,
                descent_detected=0)
    previous_state = 'standing'
    started = time.monotonic()
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            ok, original = cap.read()
            if not ok:
                break
            t = time.monotonic() - started if is_camera else frames / source_fps
            tick = time.monotonic()
            detections = detect_people(interpreter, inputs, outputs, original)
            blocked = [(region, until) for region, until in blocked if until > t]
            chosen = None
            if target is not None:
                matches = [d for d in detections if box_iou(target, d[0]) >= TARGET_IOU]
                if matches:
                    chosen = max(matches, key=lambda d: box_iou(target, d[0]))
                else:
                    # [수정] 넘어지는 순간에는 박스가 세로→가로로 급변해 IoU가 끊긴다.
                    # 중심이 가까우면 같은 사람으로 이어 붙여 추적을 유지한다.
                    limit = REACQUIRE_DIST_RATIO * max(target[3] - target[1], 1)
                    near = [d for d in detections if box_center_dist(target, d[0]) <= limit]
                    if near:
                        chosen = min(near, key=lambda d: box_center_dist(target, d[0]))
                    elif t - target_seen > MAX_GAP_SEC:
                        # [수정] 상태 기계는 초기화하지 않는다 — 초기화하면 직전 하강 이력이
                        # 사라져 낙상이 영영 확정되지 않는다. 시간 간격 처리는 update()가 담당.
                        target = None
                        pose_fail_since = None
            if target is None and detections:
                usable = [d for d in detections if not is_blocked(d[0], blocked, t)]
                if usable:
                    chosen = max(usable, key=lambda d: (d[0][2]-d[0][0])*(d[0][3]-d[0][1]))
            frame = original.copy()
            obs = None
            chosen_view = None   # [수정] 박스는 판정(sm.update) 뒤에 상태 색으로 그린다
            for box, score, cid, _fall_score in detections:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (140, 140, 140), 1)
            pose_fallback = False
            if (chosen is None and last_valid_box is not None and last_valid_pose_t is not None
                    and (t - last_valid_pose_t <= POSE_FALLBACK_SEC or sm.confirmed or sm.state == 'falling_candidate')):
                # 단기 검출 누락 때만 직전 사람 영역을 확장해 포즈 확인.
                bx1, by1, bx2, by2 = last_valid_box
                bw, bh = bx2 - bx1, by2 - by1
                fh, fw = original.shape[:2]
                search = (max(0, int(bx1-bw*.15)), max(0, int(by1-bh*.1)),
                          min(fw, int(bx2+bw*.15)), min(fh, int(by2+bh*.4)))
                search = last_pose_roi if last_pose_roi is not None else search
                chosen = (search, 0.0, -1, 0.0)
                pose_fallback = True
            if chosen is not None:
                box, score, cid, fall_score = chosen
                if not pose_fallback:
                    target, target_seen = box, t
                x1, y1, x2, y2 = box
                # 정상/정지 상태에서도 포즈를 추론하여 낙상 전후 관측을 유지.
                kpts = estimate_pose(pose, pose_inputs, pose_outputs, original, *box)
                # 검출 박스가 있어도 포즈 실패 시 최근 관절 영역으로 한 번 재확인.
                retried_pose = False
                if (kpts is None and not pose_fallback and last_pose_roi is not None
                        and last_valid_pose_t is not None
                        and (t-last_valid_pose_t <= POSE_FALLBACK_SEC or sm.confirmed or sm.state == 'falling_candidate')):
                    kpts = estimate_pose(pose, pose_inputs, pose_outputs, original, *last_pose_roi)
                    retried_pose = kpts is not None
                # [추가] 대상 박스에서 사람 포즈가 계속 안 잡히면(TV·가구 등 오검출에 고착된 경우)
                # 그 영역을 잠시 제외하고 다른 검출로 대상을 옮긴다.
                if kpts is None and not pose_fallback:
                    if pose_fail_since is None:
                        pose_fail_since = t
                    elif t - pose_fail_since > TARGET_STUCK_SEC:
                        blocked.append((box, t + TARGET_BLOCK_SEC))
                        target, pose_fail_since = None, None
                else:
                    pose_fail_since = None
                lying, angle = None, None
                pose_aspect = 0.0
                hip = None
                shoulder = None
                if kpts is not None:
                    # 유효 포즈가 계속 보이면 확인 영역 유지. 실패하면 기존 2초 만료 적용.
                    last_valid_pose_t = t
                    if not pose_fallback and not retried_pose:
                        last_valid_box = box
                        last_pose_roi = pose_search_box(kpts, box, original.shape)
                    if retried_pose:
                        pose_fallback = True
                    visible = kpts[kpts[:, 2] >= KPT_CONF_TH]
                    if len(visible) >= 5:
                        pose_aspect = float(np.ptp(visible[:, 0]) / max(1.0, np.ptp(visible[:, 1])))
                    draw_skeleton(frame, kpts)
                    lying = is_lying_down(kpts)
                    shoulder = _kpt_center(kpts, 'l_shoulder', 'r_shoulder')
                    hip = _kpt_center(kpts, 'l_hip', 'r_hip')
                    if shoulder and hip:
                        angle = _angle_from_vertical(shoulder, hip)
                if lying is not None:
                    # [수정] 살아남은 박스의 클래스만 보지 않고, 겹치는 fall 박스의 점수도 근거로 인정
                    obs = dict(cy=float(hip[1]) if hip else (y1+y2)/2,
                               height=y2-y1, aspect=(x2-x1)/(y2-y1),
                               shoulder_y=float(shoulder[1]) if shoulder else None,
                               pose_aspect=pose_aspect, pose_fallback=pose_fallback,
                               angle=angle, lying=lying,
                               fall=bool(cid == FALL_ID or fall_score > CONF_TH))
                chosen_view = (box, f'CNN:{CLASS_NAMES.get(cid, "pose-only")} {score:.2f} '
                                    f'fall:{fall_score:.2f} lying:{lying}')
            alarm = sm.update(t, obs)
            alarms += int(alarm)
            emergencies += int(sm.emergency_now)
            diag['frames_with_detection'] += int(bool(detections))
            if obs is not None:
                diag['frames_with_pose'] += 1
                diag['frames_fall_evidence'] += int(obs['fall'])
                diag['frames_lying'] += int(obs['lying'] is True)
                diag['frames_fall_and_lying'] += int(obs['fall'] and obs['lying'] is True)
            if sm.lying_run_since is not None:
                diag['longest_lying_run_sec'] = round(
                    max(diag['longest_lying_run_sec'], t - sm.lying_run_since), 2)
            if sm.state == 'falling_candidate' and previous_state != 'falling_candidate':
                diag['descent_detected'] += 1
            previous_state = sm.state
            # [수정] 판정 결과가 나온 뒤에 대상 박스를 상태 색으로 그린다
            # (예전에는 주황 고정이라 FALLEN인데도 박스가 주황으로 나왔음)
            color = STATE_COLOR.get(sm.state, (0, 220, 220))
            if chosen_view is not None:
                (bx1, by1, bx2, by2), tag = chosen_view
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(frame, f'{sm.state} | {tag}', (bx1, max(18, by1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            h, w = frame.shape[:2]
            canvas = np.zeros((h + INFO_PAD_HEIGHT, w, 3), np.uint8)
            canvas[:h] = frame
            fps = 1 / max(time.monotonic()-tick, 1e-6)
            # 영상에 확정 판정, 상대 시각, 상태를 항상 포함.
            cv2.putText(canvas, f'{sm.state.upper()}  t={t:.2f}s', (10, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
            draw_bar(canvas, h, w, f'{sm.state} {fps:.1f}fps',
                     'EMERGENCY' if sm.emergency_active else ('AUTO EVENT REC' if events.active or alarm else 'AUTO READY'), color, manual is not None)
            draw_emergency_status(canvas, sm, t)
            if sm.confirmed and obs is None and not sm.emergency_active:
                cv2.putText(canvas, 'POSE LOST - FALL ALERT RETAINED', (12, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
            if sm.emergency_active:
                frozen = None  # 이전 낙상 정지 화면이 긴급 배너를 가리지 않음
            if alarm or sm.emergency_now:
                cv2.imwrite(str(clip_path(out_dir, ext='.jpg')), canvas)
                if alarm and not sm.emergency_active:
                    frozen, freeze_until = canvas.copy(), t + FREEZE_DURATION_SEC
            # 녹화는 항상 현재 판정 화면. UI 프리즈와 추론을 분리.
            events.update(t, canvas, original, alarm or sm.emergency_now, sm.reason)
            if preview:
                preview.add(t, canvas, original)
            if manual:
                manual.add(t, canvas, original)
            log.write(json.dumps(dict(frame=frames, time=t, state=sm.state, alarm=alarm,
                                      emergency=sm.emergency_active, emergency_now=sm.emergency_now,
                                      fallen_duration=round(sm.fallen_duration, 3),
                                      observation=obs, detections=len(detections), reason=sm.reason, evidence_sec=round(sm.evidence_sec, 3),
                                      target_box=box if chosen is not None else None)) + '\n')
            frames += 1
            if not args.headless:
                display = frozen if FREEZE_ON_ALARM and frozen is not None and t < freeze_until else canvas
                cv2.imshow(WINDOW_NAME, display)
                toggle, quit_now = read_actions(cv2.waitKey(1) & 0xFF)
                if toggle:
                    if manual:
                        manual.close()
                        manual = None
                    else:
                        manual = PairedVideo(clip_path(out_dir, '_manual'))
                        manual.add(t, canvas, original)
                if quit_now:
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        events.finish(True)
        if manual:
            manual.close()
        if preview:
            preview.close()
        log.close()
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    summary = dict(frames=frames, alarms=alarms, emergencies=emergencies, emergency_sec=args.emergency_sec, elapsed_sec=time.monotonic()-started,
                   event_videos=events.completed, source=args.source, model=args.model,
                   pose_model=args.pose_model, diagnostics=diag)
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
