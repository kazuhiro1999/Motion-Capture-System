import cv2
import json
from multiprocessing import Process, Queue, Event
from queue import Empty

from utils import TimeUtil
from camera import USBCamera
from pose2d import MediapipePose
from pose3d import recover_pose_3d
from data import to_dict


def camera_process(camera_config, event, queue, cancel_event):
    camera = USBCamera(camera_config)
    camera.open()
    pose_estimator = MediapipePose()
    print(f"process {camera_config['name']} initialized")

    while not cancel_event.is_set():
        # Wait for the signal to start processing
        if not event.is_set():
            continue

        # Reset the event for the next round
        event.clear()

        frame = camera.get_image()
        if frame is not None:
            cv2.imshow(camera.name, cv2.resize(frame, dsize=(640,360)))
            keypoints2d = pose_estimator.process(frame)

            if keypoints2d is not None:
                proj_matrix = camera.camera_setting.proj_matrix
                # Put the result into the queue
                queue.put((proj_matrix, keypoints2d))
            else:
                queue.put((None, None))

        cv2.waitKey(1)

    camera.close()
    print(f"process {camera_config['name']} ended")


def capture_process(config, queue_out, cancel_event):
    # Initialize Motion Capture
    processes = []
    queues = []
    events = []
    for camera_config in config['cameras']:
        queue = Queue()
        event = Event()
        process = Process(target=camera_process, args=(camera_config, event, queue, cancel_event))

        processes.append(process)
        queues.append(queue)
        events.append(event)

        process.start()

    print("capture start")

    # Send signal to start processing
    for event in events:
        event.set()

    timestamp = TimeUtil.get_unixtime()

    # Main loop
    while not cancel_event.is_set():

        # Collect results from each process
        keypoints2d_list = []
        proj_matrices = []

        for queue in queues:
            try:
                proj_matrix, keypoints2d = queue.get(timeout=1.0)                
            except Empty:
                continue
            if keypoints2d is not None:
                proj_matrices.append(proj_matrix)
                keypoints2d_list.append(keypoints2d)

        # Send signal to start next processing
        for event in events:
            event.set()

        t_next = TimeUtil.get_unixtime()

        # 3D pose estimation 
        keypoints3d = recover_pose_3d(proj_matrices, keypoints2d_list)
        data = to_dict(timestamp, keypoints3d, MediapipePose.KEYPOINT_DICT, MediapipePose.Type)  
        queue_out.put(data)            

        timestamp = t_next

    # End subprocess
    for process in processes:
        process.join()
        process.terminate()


    cv2.destroyAllWindows()
    print("capture ended")


    