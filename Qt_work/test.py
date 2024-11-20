import cv2
import time
import numpy as np


def test_camera(resolution, fps):
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # Set video format to MJPG


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nTesting Resolution: {resolution}, FPS: {fps}")
    print(f"Actual Resolution: {actual_width}x{actual_height}, Actual FPS: {actual_fps}")

    num_frames = 300
    frame_times = []
    start = time.time()

    for i in range(num_frames):
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_times.append(time.time() - frame_start)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = time.time()
    seconds = end - start
    measured_fps = num_frames / seconds
    print(f"Measured FPS: {measured_fps:.2f}")

    # Analyze frame times
    frame_intervals = np.diff(frame_times)
    print(f"Average frame interval: {np.mean(frame_intervals):.5f} seconds")
    print(f"Std dev of frame intervals: {np.std(frame_intervals):.5f} seconds")

    # Print all camera properties
    properties = [
        "CAP_PROP_POS_MSEC", "CAP_PROP_POS_FRAMES", "CAP_PROP_POS_AVI_RATIO",
        "CAP_PROP_FRAME_WIDTH", "CAP_PROP_FRAME_HEIGHT", "CAP_PROP_FPS",
        "CAP_PROP_FOURCC", "CAP_PROP_FRAME_COUNT", "CAP_PROP_FORMAT",
        "CAP_PROP_MODE", "CAP_PROP_BRIGHTNESS", "CAP_PROP_CONTRAST",
        "CAP_PROP_SATURATION", "CAP_PROP_HUE", "CAP_PROP_GAIN",
        "CAP_PROP_EXPOSURE", "CAP_PROP_CONVERT_RGB", "CAP_PROP_RECTIFICATION"
    ]
    print("\nCamera Properties:")
    for prop in properties:
        value = cap.get(getattr(cv2, prop))
        if prop == "CAP_PROP_FOURCC":
            value = "".join([chr((int(value) >> 8 * i) & 0xFF) for i in range(4)])
        print(f"{prop}: {value}")

    cap.release()
    cv2.destroyAllWindows()


# Test different resolutions and FPS
resolutions = [(1280, 720),]
fps_values = [10,25]

for resolution in resolutions:
    for fps in fps_values:
        test_camera(resolution, fps)

print("\nDiagnostic complete. Please check the output for any inconsistencies.")