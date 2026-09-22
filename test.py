"""
라즈베리파이에서 실행 — 가위바위보 프로젝트(YOLO_Project.py)와 동일한 tflite_runtime 방식.
전처리(letterbox)·후처리(박스 디코딩, NMS) 로직은 그 코드를 그대로 재사용했습니다.
ultralytics 설치가 필요 없습니다 — 이미 검증된 환경(tflite_runtime + opencv + numpy) 그대로 씁니다.

[추가: 스켈레톤(포즈) 기반 확인]
낙상 감지 모델이 "fall"로 판단한 박스에 한해서만(연산 절약을 위해) 별도의 포즈 추정
모델을 추가로 돌려, 어깨-엉덩이 축이 수직선과 얼마나 벌어져 있는지로 "정말 누운 자세인지"
한 번 더 확인합니다. 최종 판정은 POSE_CONFIRM_MODE로 CNN 결과와 조합합니다.

사전 준비
  1) 이 파일과 같은 폴더에 낙상 탐지 tflite 파일을 둘 것 (예: best_int8.tflite)
  2) [추가] 이 파일과 같은 폴더에 포즈 추정 tflite 파일을 둘 것 (예: yolo11n-pose_int8.tflite)
     - 아직 없다면 Colab에서:
         from ultralytics import YOLO
         YOLO("yolo11n-pose.pt").export(format="tflite", int8=True, imgsz=POSE_IMG_SIZE)
       COCO로 사전학습된 모델이라 낙상 데이터셋으로 재학습할 필요는 없습니다 —
       사람 자세(키포인트)는 범용으로 잘 잡아냅니다.
     - 처음 테스트할 때는 int8=False(float32)로 먼저 export해서 동작부터 확인하고,
       이후 int8로 바꿔 속도를 올리는 순서를 추천합니다.
  3) 아래 CLASS_NAMES를 Colab data.yaml의 names 순서와 반드시 맞출 것
     (확인법: Colab에서 `print(yaml.safe_load(open('data.yaml'))['names'])`)
     순서가 다르면 fall과 non-fall이 뒤바뀐 채로 동작합니다.

실행: python3 test.py
종료: 영상 창에서 q
"""
import time
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from state_machine import FallStateMachine

# ---- 설정 -------------------------------------------------
MODEL_PATH = "best_int8.tflite"
IMG_SIZE = 320
CONF_TH = 0.45                    # 신뢰도 임계값
IOU_TH = 0.45
CONFIRM_SEC = 1.0                 # 상태 기계: 낙상 지속 시간 (초)
MOTION_TH = 3.0                   # [추가] 움직임 감지 임계값 (숫자가 낮을수록 민감)
INFO_PAD_HEIGHT = 60              # [추가] 하단 정보 표시용 검은 여백 높이(px)
DEBUG = False                     # [추가] 진단용 로그 on/off — 프레임마다 검출/모션/포즈 정보 출력

# [추가] 낙상 확정(alarm) 순간을 놓치지 않기 위한 기능 ----------------------
FREEZE_ON_ALARM = True            # 낙상이 확정되면 화면을 잠깐 정지시켜서 눈으로 확인할 시간을 줌
FREEZE_DURATION_SEC = 3.0         # 정지 상태를 유지할 시간(초)
SAVE_ALARM_SNAPSHOT = True        # 낙상 확정 순간의 화면을 이미지 파일로도 저장
RESULT_DIR = "results"            # 캡처 이미지를 저장할 폴더 (없으면 자동 생성)

# [추가: 포즈] ------------------------------------------------
POSE_MODEL_PATH = "yolo11n-pose_int8.tflite"  # 포즈 추정 모델 — 별도로 변환해서 준비 필요
POSE_IMG_SIZE = 192               # 포즈 모델 입력 크기. 변환(export) 시 imgsz와 반드시 동일해야 함
POSE_CONF_TH = 0.5                # 크롭 안에서 "사람"으로 인정할 신뢰도 임계값
KPT_CONF_TH = 0.4                 # 개별 키포인트(관절점) 신뢰도 임계값 — 이보다 낮으면 안 보이는 부위로 간주
LYING_ANGLE_TH_DEG = 55           # 어깨-엉덩이 축이 수직선과 이 각도 이상 벌어지면 "누운 자세"로 판단
POSE_PAD_RATIO = 0.15             # 박스를 크롭할 때 여유를 두는 비율 (사람이 잘리지 않도록)
POSE_CONFIRM_MODE = "and"         # "and": 포즈가 "안 누움"으로 확인되면 CNN의 fall 판정을 취소(오탐 감소)
                                   # "or" : CNN이 fall이 아니어도 포즈만으로 누운 자세면 fall로 승격(민감도 증가)

CLASS_NAMES = {0: "fall", 1: "non-fall"}
FALL_ID = [k for k, v in CLASS_NAMES.items() if v == "fall"][0]
BOX_COLOR = {
    "fall": (0, 0, 255),       # 빨강 (낙상)
    "non-fall": (0, 255, 0),   # 초록 (정상)
    "static": (128, 128, 128)  # 회색 (가만히 있는 물체/배경)
}

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


def draw_hud(canvas, h, text_lines, color):
    for i, line in enumerate(text_lines):
        cv2.putText(canvas, line, (10, h + 22 + i * 23),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)


# [추가: 포즈] ---------------------------------------------------------------
def load_pose_interpreter():
    interp = tflite.Interpreter(model_path=POSE_MODEL_PATH)
    interp.allocate_tensors()
    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    print("[pose] input :", in_d)
    print("[pose] output:", out_d)
    return interp, in_d, out_d


def estimate_pose(interp, in_d, out_d, frame, x1, y1, x2, y2):
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

    in_type = in_d[0]["dtype"]
    if in_type in (np.int8, np.uint8):
        scale, zero_point = in_d[0]["quantization"]
        if scale != 0:
            img_norm = img_resized.astype(np.float32) / scale + zero_point
        else:
            img_norm = img_resized.astype(np.float32)
        img = img_norm.astype(in_type)
    else:
        img = img_resized.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    if img.shape[-1] == 3:
        img = np.transpose(img, (0, 3, 1, 2))

    interp.set_tensor(in_d[0]["index"], img)
    interp.invoke()

    raw = interp.get_tensor(out_d[0]["index"])
    out_type = out_d[0]["dtype"]
    if out_type in (np.int8, np.uint8):
        scale, zero_point = out_d[0]["quantization"]
        if scale != 0:
            raw = (raw.astype(np.float32) - zero_point) * scale

    if raw.ndim == 3:
        raw = raw[0]
    if raw.shape[0] < raw.shape[1]:
        raw = raw.transpose()   # (N, 56) = [cx,cy,w,h, person_conf, (kx,ky,kconf)*17]

    if raw.shape[0] == 0:
        return None

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


def is_lying_down(keypoints):
    """
    어깨 중심 - 엉덩이 중심 벡터가 수직선과 이루는 각도로 누운 자세 여부 판단.
    필요한 키포인트(양 어깨, 양 엉덩이) 신뢰도가 낮으면 None(판단 보류) 반환.
    """
    pts = [keypoints[KPT["l_shoulder"]], keypoints[KPT["r_shoulder"]],
           keypoints[KPT["l_hip"]], keypoints[KPT["r_hip"]]]
    if any(p[2] < KPT_CONF_TH for p in pts):
        return None

    l_sh, r_sh, l_hip, r_hip = pts
    shoulder_c = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    hip_c = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

    dx = hip_c[0] - shoulder_c[0]
    dy = hip_c[1] - shoulder_c[1]
    angle_from_vertical = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))

    return angle_from_vertical > LYING_ANGLE_TH_DEG


def draw_skeleton(frame, keypoints, color=(0, 255, 255)):
    for x, y, c in keypoints:
        if c >= KPT_CONF_TH:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    for a, b in SKELETON_EDGES:
        if keypoints[a][2] >= KPT_CONF_TH and keypoints[b][2] >= KPT_CONF_TH:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pt1, pt2, color, 2)
# -----------------------------------------------------------------------------


def main():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]
    input_type = input_details[0]["dtype"]
    input_scale_zero = input_details[0]["quantization"]
    print("[fall-model] input :", input_details)   # [추가] DEBUG 여부와 무관하게 시작 시 한 번만 출력
    print("[fall-model] output:", output_details)

    # [추가: 포즈] 포즈 추정 모델도 함께 로드
    pose_interp, pose_in_d, pose_out_d = load_pose_interpreter()

    # [추가] 캡처 저장 폴더 준비
    os.makedirs(RESULT_DIR, exist_ok=True)

    sm = FallStateMachine(confirm_sec=CONFIRM_SEC)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    prev_t = time.time()
    prev_gray = None  # 움직임 비교를 위한 이전 프레임 저장용

    # [추가] 프리즈(화면 정지) 상태 관리용 변수
    frozen_canvas = None
    freeze_until = 0.0

    while cap.isOpened():
        loop_now = time.time()

        # ---- [추가] 프리즈 중이면 정지 화면만 계속 보여주고 새 프레임 처리는 건너뜀 ----
        if frozen_canvas is not None:
            if loop_now < freeze_until:
                cv2.imshow("fall detection", frozen_canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            else:
                frozen_canvas = None
                if DEBUG:
                    print("[DEBUG] freeze 해제 — 실시간 모니터링 재개")

        ok, frame = cap.read()
        if not ok:
            break

        frame_h, frame_w, _ = frame.shape

        # ---- 움직임 감지를 위한 그레이스케일 변환 ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 첫 프레임인 경우 비교 대상 설정
        if prev_gray is None:
            prev_gray = gray
            continue

        # 현재 프레임과 이전 프레임의 차이 계산 (움직임 추출)
        frame_diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray.copy()  # 현재 프레임을 다음 비교를 위해 저장

        # ---- 모델 전처리 ----
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

        if input_type == np.int8 or input_type == np.uint8:
            scale, zero_point = input_scale_zero
            if scale != 0:
                img_norm = img_resized.astype(np.float32) / scale + zero_point
            else:
                img_norm = img_resized.astype(np.float32)
            img = img_norm.astype(input_type)
        else:
            img = img_resized.astype(np.float32) / 255.0

        img = np.expand_dims(img, axis=0)
        if img.shape[-1] == 3:
            img = np.transpose(img, (0, 3, 1, 2))

        interpreter.set_tensor(input_index, img)
        interpreter.invoke()

        raw = interpreter.get_tensor(output_index)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.shape[0] < raw.shape[1]:
            raw = raw.transpose()

        # ---- 후처리: 신뢰도 필터 + NMS ----
        class_scores = raw[:, 4:]
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        keep_mask = confidences > CONF_TH
        filtered = raw[keep_mask]
        scores = confidences[keep_mask]
        classes = class_ids[keep_mask]

        if len(filtered) > 0:
            cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            boxes = np.stack([
                (cx - w / 2) * IMG_SIZE,
                (cy - h / 2) * IMG_SIZE,
                w * IMG_SIZE,
                h * IMG_SIZE
            ], axis=-1)
        else:
            boxes = np.array([])

        label = "non-fall"
        found_fall = False

        if len(boxes) > 0:
            keep = cv2.dnn.NMSBoxesBatched(boxes.tolist(), scores.tolist(), classes.tolist(),
                                         score_threshold=CONF_TH, nms_threshold=IOU_TH)
            if DEBUG:
                print(f"[DEBUG] 이번 프레임 검출 수: {len(keep)}")
            if len(keep) > 0:
                scale_x = frame_w / IMG_SIZE
                scale_y = frame_h / IMG_SIZE

                for i in keep:
                    x, y, bw, bh = boxes[i]

                    x1 = int(np.clip(x * scale_x, 0, frame_w))
                    y1 = int(np.clip(y * scale_y, 0, frame_h))
                    x2 = int(np.clip((x + bw) * scale_x, 0, frame_w))
                    y2 = int(np.clip((y + bh) * scale_y, 0, frame_h))

                    # ---- 핵심: 바운딩 박스 내부의 '움직임 양' 검사 ----
                    # 박스 영역 안의 픽셀 변화량 평균 계산
                    box_diff = frame_diff[y1:y2, x1:x2]
                    if box_diff.size > 0:
                        mean_motion = np.mean(box_diff)
                    else:
                        mean_motion = 0

                    current_class_id = int(classes[i])
                    current_label = CLASS_NAMES[current_class_id]

                    if DEBUG:
                        print(f"[DEBUG] box=({x1},{y1},{x2},{y2}) motion={mean_motion:.1f} "
                              f"label={current_label} score={scores[i]:.3f}")

                    # 움직임이 임계값(MOTION_TH)보다 작으면 가만히 있는 사물(티비 등)로 판단해 무시
                    if mean_motion < MOTION_TH:
                        if DEBUG:
                            print(f"[DEBUG] static 처리 (motion {mean_motion:.1f} < {MOTION_TH})")
                        color = BOX_COLOR["static"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame, f"static {scores[i]*100:.0f}%", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)
                        continue

                    # [추가: 포즈] CNN이 fall로 판단한 박스에 한해서만 포즈로 한 번 더 확인
                    # (연산 절약 — non-fall/static 박스까지 매번 포즈를 돌리지 않음)
                    pose_lying = None
                    if current_label == "fall":
                        keypoints = estimate_pose(pose_interp, pose_in_d, pose_out_d,
                                                   frame, x1, y1, x2, y2)
                        if keypoints is not None:
                            draw_skeleton(frame, keypoints)
                            pose_lying = is_lying_down(keypoints)
                            if DEBUG:
                                print(f"[DEBUG] pose_lying={pose_lying}")
                        elif DEBUG:
                            print("[DEBUG] pose: 크롭에서 사람 미검출/신뢰도 부족 — CNN 판정만 사용")

                    is_fall_final = (current_label == "fall")
                    if pose_lying is not None:
                        if POSE_CONFIRM_MODE == "and":
                            is_fall_final = is_fall_final and pose_lying
                        elif POSE_CONFIRM_MODE == "or":
                            is_fall_final = is_fall_final or pose_lying
                        if DEBUG:
                            print(f"[DEBUG] 최종 판정: is_fall_final={is_fall_final} "
                                  f"(mode={POSE_CONFIRM_MODE})")

                    # 움직임이 있는 진짜 사람/물체인 경우에만 정상 판정 진행
                    display_label = "fall" if is_fall_final else current_label
                    color = BOX_COLOR.get(display_label, (255, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    tag = f"{display_label} {scores[i]*100:.0f}%"
                    if pose_lying is not None:
                        tag += " (pose-lying)" if pose_lying else " (pose-stand)"
                    cv2.putText(frame, tag, (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2)

                    if is_fall_final:
                        found_fall = True

                if found_fall:
                    label = "fall"

        # ---- 상태 기계 업데이트 ----
        now = time.time()
        alarm = sm.update(label, now)
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        # ---- 표시용 캔버스: 프레임 아래에 검은 여백을 붙여서 정보 표시 ----
        canvas = np.zeros((frame_h + INFO_PAD_HEIGHT, frame_w, 3), dtype=np.uint8)
        canvas[:frame_h, :] = frame

        hud_color = (0, 0, 255) if sm.state == "fallen" else (0, 255, 255)
        draw_hud(
            canvas, frame_h,
            [f"state:{sm.state}  fps:{fps:.1f}"],
            hud_color,
        )

        if alarm:
            print(f"[ALARM] 낙상 확정  t={now:.1f}s")

            # [추가] 확정 순간의 화면을 파일로 저장
            if SAVE_ALARM_SNAPSHOT:
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(RESULT_DIR, f"fall_{ts}.jpg")
                cv2.imwrite(save_path, canvas)
                if DEBUG:
                    print(f"[DEBUG] 캡처 저장: {save_path}")

            # [추가] 화면을 잠깐 정지시켜 눈으로 확인할 시간을 줌
            if FREEZE_ON_ALARM:
                frozen_canvas = canvas.copy()
                freeze_until = now + FREEZE_DURATION_SEC
                if DEBUG:
                    print(f"[DEBUG] freeze 시작 — {FREEZE_DURATION_SEC:.1f}초간 화면 고정")

        cv2.imshow("fall detection", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()