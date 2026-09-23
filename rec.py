"""
record_clip.py — 비교 실험용 영상 녹화기

화면 아래 버튼(또는 키보드)으로 녹화를 시작/종료하고, 종료하는 순간 파일로 저장됩니다.
저장된 영상은 test.py / test_pose_first.py / test_pose_only.py에서
  cv2.VideoCapture(0)  →  cv2.VideoCapture("clips/clip_....avi")
로 바꿔 그대로 넣으면, 세 구조가 완전히 같은 프레임을 보고 판단하게 됩니다.

조작
  [START REC] 버튼 클릭  또는  r / 스페이스 키   → 녹화 시작
  [STOP & SAVE] 버튼 클릭 또는  r / 스페이스 키   → 녹화 종료 + 자동 저장
  [QUIT] 버튼 클릭 또는 q 키                      → 프로그램 종료
                                                    (녹화 중이면 저장하고 종료)
  한 번 실행해서 여러 개를 이어 찍을 수 있습니다 — 멈췄다 다시 시작하면 새 파일이 됩니다.

저장 위치: clips/clip_날짜_시각.avi
화면에 그려지는 버튼·글자는 저장되지 않습니다 (원본 프레임만 기록).
재생 길이는 실제 촬영 시간과 같게 맞춰집니다 — 루프 속도가 변해도 구간별로 배속되지 않습니다.

실행: python3 record_clip.py
"""
import os
import time

import cv2
import numpy as np

# ---- 설정 ----------------------------------------------------------------
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480     # 판정 스크립트들과 같은 해상도로 맞춰 둘 것
CLIP_DIR = "clips"
FOURCC = "MJPG"             # 라즈베리파이에서 가장 무난한 조합 (MJPG + .avi)
EXT = ".avi"                # mp4로 저장하고 싶으면 FOURCC="mp4v", EXT=".mp4"
REC_TARGET_FPS = 0          # 저장 파일의 fps. 0이면 자동(카메라 측정값과 REC_FPS_CAP 중 작은 값)
REC_FPS_CAP = 15            # 자동일 때의 상한 — 높을수록 파일이 커짐
WARMUP_FRAMES = 20          # 카메라 속도 측정에 쓸 프레임 수
REC_MAX_FILL_SEC = 1.0      # [속도 보정] 루프가 오래 멈췄을 때 한 번에 채워 넣을 최대 길이(초)
BAR_H = 70                  # 버튼이 들어갈 아래쪽 검은 여백 높이(px)
WINDOW_NAME = "clip recorder"

BUTTONS = {}                # 이름 -> (x1, y1, x2, y2)
_click = None               # 마지막 클릭 좌표


def on_mouse(event, x, y, flags, param):
    global _click
    if event == cv2.EVENT_LBUTTONDOWN:
        _click = (x, y)


def clicked(name, pos):
    if pos is None or name not in BUTTONS:
        return False
    x1, y1, x2, y2 = BUTTONS[name]
    return x1 <= pos[0] <= x2 and y1 <= pos[1] <= y2


def measure_fps(cap, n):
    """카메라가 실제로 몇 fps로 들어오는지 측정 (VideoWriter에 넣을 값)."""
    t0 = time.time()
    got = 0
    for _ in range(n):
        ok, _frame = cap.read()
        if not ok:
            break
        got += 1
    dt = time.time() - t0
    if got == 0 or dt <= 0:
        return 15.0
    # 측정값이 터무니없이 크거나 작으면 저장 파일의 타임스탬프가 깨지므로 범위를 제한
    return float(np.clip(got / dt, 1.0, 60.0))


def draw_ui(canvas, frame_h, frame_w, recording, elapsed, frames, status_text, blink):
    """아래 막대에 버튼과 상태를 그림. BUTTONS 좌표도 여기서 갱신."""
    bar_y = frame_h
    cv2.rectangle(canvas, (0, bar_y), (frame_w, bar_y + BAR_H), (0, 0, 0), -1)

    btn_w, btn_h = 170, 40
    by1 = bar_y + (BAR_H - btn_h) // 2
    by2 = by1 + btn_h

    # 녹화 시작/중지 버튼
    rx1 = 10
    rx2 = rx1 + btn_w
    BUTTONS["rec"] = (rx1, by1, rx2, by2)
    rec_color = (0, 0, 220) if not recording else (0, 140, 220)
    rec_label = "START REC" if not recording else "STOP & SAVE"
    cv2.rectangle(canvas, (rx1, by1), (rx2, by2), rec_color, -1)
    cv2.putText(canvas, rec_label, (rx1 + 12, by2 - 13),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)

    # 종료 버튼
    qx1 = rx2 + 12
    qx2 = qx1 + 90
    BUTTONS["quit"] = (qx1, by1, qx2, by2)
    cv2.rectangle(canvas, (qx1, by1), (qx2, by2), (70, 70, 70), -1)
    cv2.putText(canvas, "QUIT", (qx1 + 20, by2 - 13),
                cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)

    # 상태 표시
    color = (0, 0, 255) if recording else (0, 255, 255)
    if recording:
        text = f"REC {int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}  frames:{frames}  {status_text}"
    else:
        text = status_text
    cv2.putText(canvas, text, (qx2 + 15, by2 - 13), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2)

    # 녹화 중 빨간 점 (영상 위)
    if recording and blink:
        cv2.circle(canvas, (frame_w - 25, 25), 10, (0, 0, 255), -1)


def main():
    global _click

    os.makedirs(CLIP_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    cam_fps = round(measure_fps(cap, WARMUP_FRAMES), 1)
    fps = REC_TARGET_FPS if REC_TARGET_FPS > 0 else min(cam_fps, REC_FPS_CAP)
    print(f"[rec] 카메라 속도 {cam_fps} fps / 저장 fps {fps}")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    writer = None
    clip_path = None
    rec_start = 0.0
    rec_frames = 0      # 파일에 실제로 쓴 프레임 수 (복제 포함)
    src_frames = 0      # 카메라에서 받아 처리한 프레임 수
    status_text = "READY - press START REC"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[rec] 카메라에서 프레임을 읽지 못했습니다.")
            break
        frame_h, frame_w = frame.shape[:2]

        # ---- 저장은 UI를 그리기 전의 원본 프레임으로 ----
        # [속도 보정] 루프가 초당 몇 프레임을 돌든, 파일에는 "경과 시간 × 저장 fps"만큼만
        # 들어가도록 맞춘다. 느린 구간은 같은 프레임을 복제해 채우고, 빠른 구간은 버린다.
        # (화면 그리기·창 갱신 때문에 루프 속도가 카메라 속도보다 느려지면 배속되던 문제 해결)
        if writer is not None:
            due = int((time.time() - rec_start) * fps) + 1
            n = min(due - rec_frames, max(1, int(fps * REC_MAX_FILL_SEC)))
            for _ in range(max(n, 0)):
                writer.write(frame)
                rec_frames += 1
            src_frames += 1

        now = time.time()
        canvas = np.zeros((frame_h + BAR_H, frame_w, 3), dtype=np.uint8)
        canvas[:frame_h, :] = frame
        draw_ui(canvas, frame_h, frame_w,
                recording=writer is not None,
                elapsed=now - rec_start if writer is not None else 0.0,
                frames=rec_frames,
                status_text=status_text,
                blink=int(now * 2) % 2 == 0)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF

        pos, _click = _click, None
        toggle = clicked("rec", pos) or key in (ord("r"), ord(" "))
        quit_now = clicked("quit", pos) or key == ord("q")

        # ---- 녹화 시작 ----
        if toggle and writer is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            clip_path = os.path.join(CLIP_DIR, f"clip_{ts}{EXT}")
            writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*FOURCC),
                                     fps, (frame_w, frame_h))
            if not writer.isOpened():
                writer = None
                status_text = "WRITER FAILED - check FOURCC/EXT"
                print(f"[rec] 저장 파일을 열지 못했습니다: {clip_path} "
                      f"(FOURCC={FOURCC}, EXT={EXT} 조합을 바꿔보세요)")
            else:
                rec_start = time.time()
                rec_frames = 0
                src_frames = 0
                status_text = os.path.basename(clip_path)
                print(f"[rec] 녹화 시작: {clip_path}")

        # ---- 녹화 종료 + 자동 저장 ----
        elif toggle and writer is not None:
            status_text = stop_recording(writer, clip_path, rec_start, rec_frames,
                                         src_frames, fps)
            writer = None

        if quit_now:
            break

    if writer is not None:
        stop_recording(writer, clip_path, rec_start, rec_frames, src_frames, fps)

    cap.release()
    cv2.destroyAllWindows()


def stop_recording(writer, clip_path, rec_start, rec_frames, src_frames, fps):
    duration = max(time.time() - rec_start, 1e-6)
    writer.release()

    size_mb = os.path.getsize(clip_path) / (1024 * 1024) if os.path.exists(clip_path) else 0.0
    print(f"[rec] 저장 완료: {clip_path}")
    print(f"[rec]   {rec_frames} 프레임 @ {fps} fps → 재생 {rec_frames / fps:.1f}초 "
          f"/ 실제 {duration:.1f}초 / {size_mb:.1f} MB")
    print(f"[rec]   카메라에서 받은 프레임 {src_frames}장 ({src_frames / duration:.1f} fps) "
          f"— 저장 fps에 맞춰 복제/생략해 재생 길이를 실제 시간과 맞췄습니다.")
    return f"SAVED {os.path.basename(clip_path)} ({rec_frames}f)"


if __name__ == "__main__":
    main()