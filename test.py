import threading
import cv2
from concurrent.futures import ThreadPoolExecutor

from pose2d import MediapipePose
from visalization import draw_keypoints


def test():
    cap = cv2.VideoCapture("C:/Users/esaki/ws2023/videos/0_keito_1/0_keito_1_0_0.mp4")
    estimator = MediapipePose()

    i = 0
    with ThreadPoolExecutor(max_workers=4) as executor:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            future = executor.submit(estimator.process, frame)
            keypoints2d = future.result()
            debug_image = draw_keypoints(frame, keypoints2d, estimator.KINEMATIC_TREE)
            cv2.imshow('image', debug_image)
            print(i)
            i += 1

            if cv2.waitKey(1) == 27:
                break

    cap.release()


t = threading.Thread(target=test)
t.start()