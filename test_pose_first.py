"""
[스켈레톤 우선 구조] test_pose_first.py

test.py와 판단 순서를 뒤집은 버전입니다.
  test.py            : 낙상 CNN(전체 프레임) → CNN이 fall로 본 박스만 크롭해서 포즈로 재확인
  test_pose_first.py : 포즈 모델(전체 프레임) → 사람마다 누움 판단 → 누운 후보가 있을 때만
                       낙상 CNN을 돌려서, 같은 사람의 CNN 판정으로 "거부권"만 행사

판정 규칙 (사람 한 명 = 트랙 하나 기준)
  1) 최근 FALL_WINDOW_FRAMES번의 포즈 판단 중 "누움"이 FALL_CONFIRM_FRAMES번 이상이면 낙상 후보
  2) 그동안 CNN이 같은 사람을 non-fall로 본 표가 CNN_VETO_MIN_VOTES 이상이고
     fall 표보다 많으면 취소(veto) — 화면에 보라색 "vetoed"로 표시
  3) CNN이 그 사람을 아예 못 잡은 경우(겹치는 CNN 박스 없음)는 포즈 판단을 그대로 따름
     → test.py 구조에서는 불가능했던 "CNN이 놓친 낙상을 스켈레톤이 잡는" 경우가 여기서 생김
  4) 포즈가 STAND_RESET_FRAMES번 연속 "안 누움"이면 그 사람의 누적 상태 초기화 (일어남)

누움 판단(is_lying_down)의 3지표 투표와 임계값, 움직임 게이팅, 캡처/프리즈 동작은
test.py와 동일하게 맞춰 두었습니다 — 구조 차이만 비교하기 위해서입니다.

사전 준비
  1) 포즈 모델을 전체 프레임용으로 다시 export (test.py의 192 모델은 크롭용이라 작음)
       from ultralytics import YOLO
       YOLO("yolo11n-pose.pt").export(format="tflite", int8=True, imgsz=320)
     생성된 yolo11n-pose_saved_model/yolo11n-pose_int8.tflite를
     yolo11n-pose_320_int8.tflite로 이름을 바꿔 이 파일과 같은 폴더에 둘 것
     (test.py가 쓰는 192 모델과 파일명이 겹치지 않게 하기 위함)
  2) 낙상 CNN(best_int8.tflite)과 state_machine.py도 같은 폴더에 둘 것
  3) CLASS_NAMES 순서는 test.py와 동일하게 data.yaml의 names 순서에 맞출 것
  입력 크기/레이아웃(NCHW·NHWC)/양자화는 모델 정보에서 자동으로 읽으므로 따로 맞출 필요 없음

실행: python3 test_pose_first.py
종료: 영상 창에서 q
"""
import os
import time
from collections import deque

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from state_machine import FallStateMachine

# ---- 모델 ----------------------------------------------------------------
POSE_MODEL_PATH = "yolo11n-pose_int8.tflite"  # 전체 프레임용 포즈 모델 (imgsz=320 권장)
FALL_MODEL_PATH = "best_Hfall_int8.tflite"              # 기존 낙상 CNN (보조 확인용)
NUM_THREADS = 4                                   # tflite CPU 스레드 수

# ---- 포즈: 사람 검출 + 누움 판단 (값은 test.py와 동일) ---------------------
POSE_CONF_TH = 0.5          # 사람으로 인정할 신뢰도
POSE_IOU_TH = 0.45          # 사람 박스 NMS 임계값
KPT_CONF_TH = 0.4           # 개별 키포인트 신뢰도 — 이보다 낮으면 안 보이는 부위로 간주
LYING_ANGLE_TH_DEG = 55     # [지표1] 몸통(어깨→엉덩이) 축이 수직에서 벌어진 각도
LEG_ANGLE_TH_DEG = 60       # [지표2] 다리(엉덩이→발목) 축이 수직에서 벌어진 각도
KPT_ASPECT_TH = 1.0         # [지표3] 보이는 키포인트 전체의 가로/세로 비율
LYING_VOTE_MIN = 2          # 3개 지표 중 이 개수 이상이면 "누움"

# ---- 시간 누적 판단 (프레임 단위 노이즈 완충) ------------------------------
FALL_WINDOW_FRAMES = 5      # 최근 몇 번의 포즈 판단을 볼지
FALL_CONFIRM_FRAMES = 3     # 그중 "누움"이 이 횟수 이상이면 낙상 후보
STAND_RESET_FRAMES = 3      # "안 누움"이 이만큼 연속되면 누적 상태 초기화 (일어남)

# ---- 낙상 CNN (보조: 거부권만) ---------------------------------------------
CNN_CONF_TH = 0.45
CNN_NMS_IOU_TH = 0.45
CNN_MATCH_IOU_TH = 0.3      # 포즈의 사람 박스와 이 IoU 이상 겹치는 CNN 박스를 같은 사람으로 봄
CNN_VETO_MIN_VOTES = 2      # CNN non-fall 표가 이 이상이고 fall 표보다 많으면 취소
CLASS_NAMES = {0: "fall", 1: "non-fall"}

# ---- 움직임 게이팅 (test.py와 동일) ----------------------------------------
USE_MOTION_GATE = True      # False로 바꾸면 가만히 있는 사람도 판단 (게이팅 효과 비교용)
MOTION_TH = 3.0

# ---- 사람 추적 (프레임 간 같은 사람 연결) -----------------------------------
TRACK_IOU_TH = 0.2          # 이전 프레임 박스와 이 IoU 이상 겹치면 같은 사람
TRACK_TTL_SEC = 1.0         # 이 시간 동안 안 보이면 추적 종료

# ---- 표시/캡처 (test.py와 동일) -------------------------------------------
CAM_W, CAM_H = 640, 480
CONFIRM_SEC = 1.0           # 상태 기계(HUD 표시용) 낙상 지속 시간
INFO_PAD_HEIGHT = 60
DEBUG = False
FREEZE_ON_ALARM = True
FREEZE_DURATION_SEC = 3.0
SAVE_ALARM_SNAPSHOT = True
RESULT_DIR = "results_pose_first"   # 구조별로 캡처 폴더를 분리해 비교하기 쉽게
CAPTURE_COOLDOWN_SEC = 5.0
WINDOW_NAME = "fall detection (pose-first)"

COLOR = {
    "fall": (0, 0, 255),        # 빨강: 낙상 확정
    "lying": (0, 165, 255),     # 주황: 누운 자세지만 아직 누적 중
    "vetoed": (255, 0, 255),    # 보라: 포즈는 낙상인데 CNN이 거부
    "stand": (0, 255, 0),       # 초록: 서 있음/앉아 있음
    "unknown": (0, 255, 255),   # 노랑: 키포인트 부족으로 판단 보류
    "static": (128, 128, 128),  # 회색: 움직임 없음 (판단 안 함)
}
SKELETON_COLOR = (255, 255, 0)

# COCO 17 키포인트 인덱스 & 스켈레톤 연결선
KPT = {
    "nose": 0, "l_eye": 1, "r_eye": 2, "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16,
}
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
]


# =============================================================================
# 모델 공통
# =============================================================================
class TFLiteModel:
    """tflite 모델 하나를 감싸서 입력 크기·레이아웃·양자화를 모델 정보에서 자동으로 맞춘다."""

    def __init__(self, path, name):
        if not os.path.exists(path):
            raise SystemExit(f"[{name}] 모델 파일이 없습니다: {path}")
        self.name = name
        self.interp = tflite.Interpreter(model_path=path, num_threads=NUM_THREADS)
        self.interp.allocate_tensors()
        self.in_d = self.interp.get_input_details()[0]
        self.out_d = self.interp.get_output_details()[0]

        shape = self.in_d["shape"]
        self.nchw = int(shape[1]) == 3
        if self.nchw:
            self.in_h, self.in_w = int(shape[2]), int(shape[3])
        else:
            self.in_h, self.in_w = int(shape[1]), int(shape[2])

        print(f"[{name}] input : shape={[int(v) for v in shape]} dtype={np.dtype(self.in_d['dtype']).name} "
              f"layout={'NCHW' if self.nchw else 'NHWC'} quant={self.in_d['quantization']}")
        print(f"[{name}] output: shape={[int(v) for v in self.out_d['shape']]} "
              f"dtype={np.dtype(self.out_d['dtype']).name}")

    def run(self, img_rgb):
        """img_rgb: (in_h, in_w, 3) uint8 RGB. 반환: (후보 수, 채널) float32."""
        x = img_rgb.astype(np.float32) / 255.0
        dtype = self.in_d["dtype"]
        if dtype in (np.int8, np.uint8):
            # [0,1]로 정규화한 뒤 양자화해야 맞음 (float 입력 모델이면 이 분기는 안 탐)
            scale, zero_point = self.in_d["quantization"]
            if scale > 0:
                x = x / scale + zero_point
            info = np.iinfo(dtype)
            x = np.clip(np.round(x), info.min, info.max)
        x = np.expand_dims(x.astype(dtype), 0)
        if self.nchw:
            x = np.transpose(x, (0, 3, 1, 2))

        self.interp.set_tensor(self.in_d["index"], x)
        self.interp.invoke()

        raw = self.interp.get_tensor(self.out_d["index"])
        if self.out_d["dtype"] in (np.int8, np.uint8):
            scale, zero_point = self.out_d["quantization"]
            raw = (raw.astype(np.float32) - zero_point) * (scale if scale > 0 else 1.0)
        raw = raw.astype(np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.T
        return raw


def letterbox(img, new_w, new_h, color=114):
    """비율을 유지한 채 (new_w, new_h)에 맞추고 남는 곳은 회색으로 채움 (Ultralytics 학습 방식과 동일)."""
    h, w = img.shape[:2]
    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (new_w - nw) // 2, (new_h - nh) // 2
    out = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, r, pad_x, pad_y


def _coord_scale(cand, model):
    """출력 좌표가 0~1로 정규화돼 있으면 입력 크기 픽셀로 되돌리는 배율."""
    if cand[:, :4].max() <= 2.0:
        return float(model.in_w), float(model.in_h)
    return 1.0, 1.0


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# =============================================================================
# 1단계: 포즈 모델 — 전체 프레임에서 사람 + 17 키포인트
# =============================================================================
def detect_people(pose, frame):
    """반환: [(box(x1,y1,x2,y2), score, keypoints(17,3))] — 모두 원본 프레임 좌표."""
    fh, fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lb, r, pad_x, pad_y = letterbox(rgb, pose.in_w, pose.in_h)
    raw = pose.run(lb)

    if raw.shape[1] != 5 + 17 * 3:
        raise SystemExit(f"[pose] 출력 채널이 56이 아닙니다: {raw.shape} — 포즈 모델이 맞는지 확인하세요.")

    cand = raw[raw[:, 4] >= POSE_CONF_TH]
    if len(cand) == 0:
        return []

    sx, sy = _coord_scale(cand, pose)
    cx, cy = cand[:, 0] * sx, cand[:, 1] * sy
    w, h = cand[:, 2] * sx, cand[:, 3] * sy
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)

    keep = cv2.dnn.NMSBoxes(boxes.tolist(), cand[:, 4].tolist(), POSE_CONF_TH, POSE_IOU_TH)
    people = []
    for k in np.array(keep).flatten():
        k = int(k)
        bx, by, bw, bh = boxes[k]
        x1 = int(np.clip((bx - pad_x) / r, 0, fw))
        y1 = int(np.clip((by - pad_y) / r, 0, fh))
        x2 = int(np.clip((bx + bw - pad_x) / r, 0, fw))
        y2 = int(np.clip((by + bh - pad_y) / r, 0, fh))
        if x2 <= x1 or y2 <= y1:
            continue

        kpts = cand[k, 5:].reshape(17, 3).copy()
        kpts[:, 0] = (kpts[:, 0] * sx - pad_x) / r
        kpts[:, 1] = (kpts[:, 1] * sy - pad_y) / r
        people.append(((x1, y1, x2, y2), float(cand[k, 4]), kpts))
    return people


# =============================================================================
# 3단계: 낙상 CNN — 누운 후보가 있을 때만 실행 (전처리는 test.py와 동일한 단순 resize)
# =============================================================================
def detect_fall_cnn(cnn, frame):
    """반환: [(box(x1,y1,x2,y2), label, score)] — 원본 프레임 좌표."""
    fh, fw = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raw = cnn.run(cv2.resize(rgb, (cnn.in_w, cnn.in_h)))

    if raw.shape[1] != 4 + len(CLASS_NAMES):
        raise SystemExit(f"[fall-cnn] 출력 채널 수가 CLASS_NAMES와 맞지 않습니다: {raw.shape}")

    class_scores = raw[:, 4:]
    conf = class_scores.max(axis=1)
    cls = class_scores.argmax(axis=1)
    m = conf > CNN_CONF_TH
    if not m.any():
        return []
    cand, conf, cls = raw[m], conf[m], cls[m]

    sx, sy = _coord_scale(cand, cnn)
    cx, cy = cand[:, 0] * sx, cand[:, 1] * sy
    w, h = cand[:, 2] * sx, cand[:, 3] * sy
    boxes = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)

    keep = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(), cls.tolist(),
                                   CNN_CONF_TH, CNN_NMS_IOU_TH)
    scale_x, scale_y = fw / cnn.in_w, fh / cnn.in_h
    dets = []
    for k in np.array(keep).flatten():
        k = int(k)
        bx, by, bw, bh = boxes[k]
        x1 = int(np.clip(bx * scale_x, 0, fw))
        y1 = int(np.clip(by * scale_y, 0, fh))
        x2 = int(np.clip((bx + bw) * scale_x, 0, fw))
        y2 = int(np.clip((by + bh) * scale_y, 0, fh))
        dets.append(((x1, y1, x2, y2), CLASS_NAMES[int(cls[k])], float(conf[k])))
    return dets


def match_cnn(person_box, cnn_dets):
    """사람 박스와 충분히 겹치는 CNN 박스 중 신뢰도가 가장 높은 것의 라벨. 없으면 None.
    (NMS가 클래스별로 돌아서 한 사람에 fall/non-fall 박스가 둘 다 남을 수 있으므로 신뢰도로 고름)"""
    best_label, best_score = None, -1.0
    for cbox, clabel, cscore in cnn_dets:
        if iou(person_box, cbox) >= CNN_MATCH_IOU_TH and cscore > best_score:
            best_label, best_score = clabel, cscore
    return best_label


# =============================================================================
# 사람 추적 — 누적 판단을 "사람별로" 하기 위함 (test.py의 전역 streak 문제 해결)
# =============================================================================
class Track:
    _next_id = 0

    def __init__(self, box, now):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.last_seen = now
        self.reset()

    def reset(self):
        self.history = deque(maxlen=FALL_WINDOW_FRAMES)  # 최근 포즈 판단 (True=누움)
        self.stand_streak = 0
        self.cnn_fall_votes = 0
        self.cnn_nonfall_votes = 0
        self.alarmed = False   # 이 사람의 이번 낙상을 이미 캡처했는지


def update_tracks(tracks, people, now):
    """이번 프레임의 사람들을 기존 트랙과 IoU로 짝지음. 반환: people과 같은 순서의 Track 리스트."""
    pairs = []
    for ti, t in enumerate(tracks):
        for pi, (box, _, _) in enumerate(people):
            v = iou(t.box, box)
            if v >= TRACK_IOU_TH:
                pairs.append((v, ti, pi))
    pairs.sort(reverse=True)

    assigned = [None] * len(people)
    used = set()
    for _, ti, pi in pairs:
        if ti in used or assigned[pi] is not None:
            continue
        assigned[pi] = tracks[ti]
        used.add(ti)

    for pi, (box, _, _) in enumerate(people):
        if assigned[pi] is None:
            assigned[pi] = Track(box, now)
            tracks.append(assigned[pi])
        assigned[pi].box = box
        assigned[pi].last_seen = now

    tracks[:] = [t for t in tracks if now - t.last_seen <= TRACK_TTL_SEC]
    return assigned


# =============================================================================
# 누움 판단 — test.py와 동일한 3지표 투표
# =============================================================================
def _kpt_center(keypoints, name_a, name_b):
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
    """True(누움) / False(안 누움) / None(계산 가능한 지표가 2개 미만 → 판단 보류)."""
    votes, checks = 0, 0
    info = []

    shoulder_c = _kpt_center(keypoints, "l_shoulder", "r_shoulder")
    hip_c = _kpt_center(keypoints, "l_hip", "r_hip")
    ankle_c = _kpt_center(keypoints, "l_ankle", "r_ankle")

    if shoulder_c and hip_c:
        torso = _angle_from_vertical(shoulder_c, hip_c)
        checks += 1
        votes += int(torso > LYING_ANGLE_TH_DEG)
        info.append(f"torso={torso:.0f}")

    if hip_c and ankle_c:
        leg = _angle_from_vertical(hip_c, ankle_c)
        checks += 1
        votes += int(leg > LEG_ANGLE_TH_DEG)
        info.append(f"leg={leg:.0f}")

    vis = keypoints[keypoints[:, 2] >= KPT_CONF_TH]
    if len(vis) >= 5:
        aspect = float(np.ptp(vis[:, 0])) / (float(np.ptp(vis[:, 1])) + 1e-6)
        checks += 1
        votes += int(aspect > KPT_ASPECT_TH)
        info.append(f"aspect={aspect:.2f}")

    if DEBUG:
        print(f"[DEBUG]   pose 지표: {' '.join(info) or '(계산 불가)'}  votes={votes}/{checks}")

    if checks < 2:
        return None
    return bool(votes >= LYING_VOTE_MIN)


# =============================================================================
# 그리기
# =============================================================================
def draw_skeleton(frame, keypoints, color=SKELETON_COLOR):
    for x, y, c in keypoints:
        if c >= KPT_CONF_TH:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    for a, b in SKELETON_EDGES:
        if keypoints[a][2] >= KPT_CONF_TH and keypoints[b][2] >= KPT_CONF_TH:
            cv2.line(frame, (int(keypoints[a][0]), int(keypoints[a][1])),
                     (int(keypoints[b][0]), int(keypoints[b][1])), color, 2)


def draw_box(frame, box, color, text, thick=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    cv2.putText(frame, text, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_PLAIN, 1.1, color, thick)


def draw_hud(canvas, h, text_lines, color):
    for i, line in enumerate(text_lines):
        cv2.putText(canvas, line, (10, h + 22 + i * 23), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)


# =============================================================================
# 메인 루프
# =============================================================================
def main():
    pose = TFLiteModel(POSE_MODEL_PATH, "pose")
    cnn = TFLiteModel(FALL_MODEL_PATH, "fall-cnn")
    if min(pose.in_w, pose.in_h) < 256:
        print(f"[경고] 포즈 모델 입력이 {pose.in_w}x{pose.in_h}입니다. 전체 프레임을 이 크기로 줄이면 "
              f"사람이 작아져 키포인트가 부정확해질 수 있으니 imgsz=320으로 다시 export하는 걸 권장합니다.")

    os.makedirs(RESULT_DIR, exist_ok=True)
    sm = FallStateMachine(confirm_sec=CONFIRM_SEC)

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./clips/clip_20260923_092657.avi")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    tracks = []
    prev_t = time.time()
    prev_gray = None
    frozen_canvas = None
    freeze_until = 0.0
    capture_cooldown_until = 0.0

    while cap.isOpened():
        loop_now = time.time()

        # ---- 프리즈 중이면 정지 화면만 보여줌 (test.py와 동일) ----
        if frozen_canvas is not None:
            if loop_now < freeze_until:
                cv2.imshow(WINDOW_NAME, frozen_canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            frozen_canvas = None
            prev_gray = None  # 몇 초 전 프레임과 차분하면 전체가 움직임으로 잡히므로 기준 재설정
            capture_cooldown_until = loop_now + CAPTURE_COOLDOWN_SEC
            for t in tracks:
                t.last_seen = loop_now  # 프리즈 동안 끊긴 추적을 이어 붙임 (같은 낙상 재캡처 방지)
            if DEBUG:
                print("[DEBUG] freeze 해제 — 실시간 모니터링 재개")

        ok, frame = cap.read()
        if not ok:
            break
        frame_h, frame_w = frame.shape[:2]

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if prev_gray is None:
            prev_gray = gray
            continue
        frame_diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray

        # ---- 1단계: 포즈 모델로 사람 + 키포인트 (전체 프레임) ----
        people = detect_people(pose, frame)
        t_det = time.time()
        assigned = update_tracks(tracks, people, t_det)
        if DEBUG:
            print(f"[DEBUG] 사람 수: {len(people)}")

        # ---- 2단계: 사람마다 움직임 확인 + 누움 판단 ----
        judged = []  # [box, score, kpts, track, moving, lying]
        for (box, score, kpts), trk in zip(people, assigned):
            moving = True
            if USE_MOTION_GATE:
                x1, y1, x2, y2 = box
                roi = frame_diff[y1:y2, x1:x2]
                motion = float(np.mean(roi)) if roi.size > 0 else 0.0
                moving = motion >= MOTION_TH
                if DEBUG:
                    print(f"[DEBUG] #{trk.id} box={box} motion={motion:.1f} "
                          f"{'' if moving else '→ static'}")
            lying = is_lying_down(kpts) if moving else None
            judged.append((box, score, kpts, trk, moving, lying))

        # ---- 3단계: 누운 후보가 있을 때만 낙상 CNN 실행 (그리기 전의 깨끗한 프레임으로) ----
        cnn_ran = any(j[5] is True for j in judged)
        cnn_dets = detect_fall_cnn(cnn, frame) if cnn_ran else []
        if DEBUG and cnn_ran:
            print(f"[DEBUG] CNN 실행: {[(d[1], round(d[2], 2)) for d in cnn_dets]}")

        # ---- 4단계: 사람별 누적 → 최종 판정 → 그리기 ----
        any_fall = False
        capture_tracks = []
        for box, score, kpts, trk, moving, lying in judged:
            if not moving:
                draw_box(frame, box, COLOR["static"], f"#{trk.id} static", 1)
                continue

            draw_skeleton(frame, kpts)
            cnn_label = None
            if lying is True:
                trk.history.append(True)
                trk.stand_streak = 0
                cnn_label = match_cnn(box, cnn_dets)
                if cnn_label == "fall":
                    trk.cnn_fall_votes += 1
                elif cnn_label == "non-fall":
                    trk.cnn_nonfall_votes += 1
            elif lying is False:
                trk.history.append(False)
                trk.stand_streak += 1
                if trk.stand_streak >= STAND_RESET_FRAMES:
                    trk.reset()   # 일어남 → 누적 상태 초기화
            # lying is None(키포인트 부족): 누적 상태 그대로 유지

            lying_count = sum(trk.history)
            candidate = lying_count >= FALL_CONFIRM_FRAMES
            vetoed = (trk.cnn_nonfall_votes >= CNN_VETO_MIN_VOTES
                      and trk.cnn_nonfall_votes > trk.cnn_fall_votes)
            is_fall = candidate and not vetoed

            if is_fall:
                status = "fall"
            elif candidate and vetoed:
                status = "vetoed"
            elif lying is True:
                status = "lying"
            elif lying is False:
                status = "stand"
            else:
                status = "unknown"

            tag = (f"#{trk.id} {status} {score * 100:.0f}% "
                   f"L{lying_count}/{FALL_WINDOW_FRAMES} "
                   f"cnnF{trk.cnn_fall_votes}N{trk.cnn_nonfall_votes}")
            draw_box(frame, box, COLOR[status], tag)

            if DEBUG:
                print(f"[DEBUG] #{trk.id} lying={lying} cnn={cnn_label} window={list(trk.history)} "
                      f"votes F{trk.cnn_fall_votes}/N{trk.cnn_nonfall_votes} → {status}")

            if is_fall:
                any_fall = True
                if not trk.alarmed:
                    capture_tracks.append(trk)

        # ---- 상태 기계 (HUD 표시용, test.py와 동일) ----
        now = time.time()
        sm.update("fall" if any_fall else "non-fall", now)
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        canvas = np.zeros((frame_h + INFO_PAD_HEIGHT, frame_w, 3), dtype=np.uint8)
        canvas[:frame_h, :] = frame

        in_cooldown = now < capture_cooldown_until
        do_capture = bool(capture_tracks) and not in_cooldown
        if do_capture:
            for t in capture_tracks:
                t.alarmed = True

        hud_lines = [f"state:{sm.state}  fps:{fps:.1f}  people:{len(people)}  "
                     f"cnn:{'run' if cnn_ran else '-'}"]
        if do_capture:
            hud_lines.append(f"FALL CAPTURED  {time.strftime('%H:%M:%S')}")
        hud_color = (0, 0, 255) if (do_capture or sm.state == "fallen") else (0, 255, 255)
        draw_hud(canvas, frame_h, hud_lines, hud_color)

        if capture_tracks and in_cooldown and DEBUG:
            print(f"[DEBUG] 낙상 조건 충족했지만 쿨다운 중 (남은 {capture_cooldown_until - now:.1f}초)")

        if do_capture:
            ids = ",".join(str(t.id) for t in capture_tracks)
            print(f"[CAPTURE] 낙상 포착  t={now:.1f}s  person=#{ids}")
            if SAVE_ALARM_SNAPSHOT:
                ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((now % 1) * 1000):03d}"
                save_path = os.path.join(RESULT_DIR, f"fall_{ts}.jpg")
                ok_write = cv2.imwrite(save_path, canvas)
                print(f"[CAPTURE] 저장 {'성공' if ok_write else '실패'}: {save_path}")
            if FREEZE_ON_ALARM:
                frozen_canvas = canvas.copy()
                freeze_until = now + FREEZE_DURATION_SEC
            else:
                capture_cooldown_until = now + CAPTURE_COOLDOWN_SEC

        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()