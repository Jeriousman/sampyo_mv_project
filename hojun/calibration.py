import cv2
import numpy as np
import glob
# 1. 체스 보드 코너 설정 (7x7 내부 코너)
chessboard_size = (7, 7)
square_size = 400  # 체스 보드의 각 사각형 크기 (단위: 임의, 예: mm)
# 2. 실제 3D 공간에서의 체스 보드 좌표 준비
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size
# 저장할 배열 초기화
obj_points = []  # 3D 점
img_points = []  # 2D 점
# 3. 체스 보드 이미지를 불러옴
images = glob.glob("chessboard_images/*.jpg")  # 캘리브레이션 이미지 경로 지정
if not images:
    print("체스 보드 이미지가 없습니다. 경로를 확인하세요.")
    exit()
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"이미지를 불러오지 못했습니다: {fname}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 체스 보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        print(f"코너를 찾았습니다: {fname}")
        obj_points.append(objp)  # 3D 점 추가
        img_points.append(corners)  # 2D 점 추가
        # 코너 그리기
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(500)
    else:
        print(f"코너를 찾지 못했습니다: {fname}")
cv2.destroyAllWindows()
# 4. 캘리브레이션 수행
if obj_points and img_points:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    # 결과 출력
    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    # 5. 결과 저장
    np.save("camera_matrix.npy", camera_matrix)
    np.save("dist_coeffs.npy", dist_coeffs)
else:
    print("코너를 검출하지 못해 캘리브레이션을 수행할 수 없습니다.")