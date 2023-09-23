import time
import cv2
import json
from multiprocessing import Process, Queue, Event
from queue import Empty

from utils import TimeUtil
from camera import CameraSetting, USBCamera
from pose2d import MediapipePose
from pose3d import recover_pose_3d
from data import to_dict
from visalization import draw_keypoints


def camera_process(camera_config, event, queue_out, cancel_event, model_complexity=1):
    camera = USBCamera(camera_config)
    camera.open()
    pose_estimator = MediapipePose(model_complexity=model_complexity)

    print(f"process {camera_config['name']} initialized")
    event.set()

    while not cancel_event.is_set():
        # Wait for the signal to start processing
        if event.is_set():
            time.sleep(0.01)
            continue

        frame = camera.get_image()
        if frame is not None:
            keypoints2d = pose_estimator.process(frame)
            debug_image = draw_keypoints(frame, keypoints2d, MediapipePose.KINEMATIC_TREE)
            cv2.imshow(camera.name, cv2.resize(debug_image, dsize=(640,360)))

        # Put the result into the queue
        queue_out.put((frame, keypoints2d))
        cv2.waitKey(1)

        # ready for next processing
        event.set()

    camera.close()
    cv2.destroyAllWindows()

    print(f"process {camera_config['name']} ended")
    return


def capture_process(config, init_event, queue_out, cancel_event, model_complexity=1):
    # Initialize Motion Capture
    processes = []
    queues = []
    events = []
    camera_settings = []
    for camera_config in config['cameras']:
        queue = Queue()
        event = Event()
        process = Process(target=camera_process, args=(camera_config, event, queue, cancel_event, model_complexity))

        processes.append(process)
        queues.append(queue)
        events.append(event)

        process.start()

        camera_setting = CameraSetting(camera_config['setting_path'])
        camera_settings.append(camera_setting)

    # Wait for the signal to start processing
    while not all(event.is_set() for event in events):
        time.sleep(0.01)  

    # All process initialized
    init_event.set()

    # Reset the event for the next round
    for event in events:
        event.clear()

    timestamp = TimeUtil.get_unixtime()

    # Main loop
    while not cancel_event.is_set():

        # Collect results from each process
        keypoints2d_list = []
        proj_matrices = []

        for i, queue in enumerate(queues):
            try:
                frame, keypoints2d = queue.get(timeout=1.0)                
            except Empty:
                continue
            if keypoints2d is not None:
                proj_matrices.append(camera_settings[i].proj_matrix)
                keypoints2d_list.append(keypoints2d)

        # Reset the event for the next processing
        for event in events:
            event.clear()

        t_next = TimeUtil.get_unixtime()

        # 3D pose estimation 
        keypoints3d = recover_pose_3d(proj_matrices, keypoints2d_list) 

        # Remove old data 
        while queue_out.qsize() > 1:
            queue_out.get_nowait()

        queue_out.put((timestamp, keypoints2d_list, keypoints3d))            

        timestamp = t_next

    # End subprocess
    for process in processes:
        process.join(timeout=1.0)
        process.terminate()

    cv2.destroyAllWindows()
    print("capture ended")
    return


    