import cv2
import datetime

# RTSP 링크 설정
rtsp_link = "rtsp://admin:sampyo123!@172.17.16.6:558/LiveChannel/0/media.smp"

# OpenCV VideoCapture 객체 생성
cap = cv2.VideoCapture(rtsp_link)
print('cap 생성 완료')

# RTSP 스트림이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("Error: Failed to open RTSP stream")
    exit()

# 캡처할 동영상의 길이 (초)
record_duration = 600  # 원하는 녹화 시간 (초)
fps = 20.0  # 프레임 레이트 설정 (RTSP 스트림의 프레임 레이트와 맞추세요)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 저장 경로 설정
now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
save_path = f"recorded_video_{current_time}.avi"

# VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정
out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

# 녹화 시작 시간
start_time = datetime.datetime.now()

while (datetime.datetime.now() - start_time).seconds < record_duration:
    ret, frame = cap.read()  # 프레임 읽기
    if ret:
        out.write(frame)  # 프레임을 비디오 파일에 쓰기
    else:
        print("Error: Failed to capture frame")
        break

# VideoCapture 및 VideoWriter 객체 해제
cap.release()
out.release()

print(f"Video recorded and saved to {save_path}")
