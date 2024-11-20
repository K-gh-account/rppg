import cv2
import time

# 打开摄像头
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 25)
cap.set(3,1280)
cap.set(4,720)


cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


print(cap.get(cv2.CAP_PROP_FPS))
# 初始化帧率计数器和时间变量
prev_time = time.time()
fps = 0
frame_count = 0

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    if not ret:
        break

    # 计算帧率
    frame_count += 1
    curr_time = time.time()
    elapsed_time = curr_time - prev_time
    if elapsed_time > 1:  # 每 1 秒更新一次帧率
        fps = frame_count / elapsed_time
        frame_count = 0
        prev_time = curr_time

    # 在帧上显示帧率
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示帧
    cv2.imshow('Frame', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()