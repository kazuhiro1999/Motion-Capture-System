import json
import os
import threading
import time
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from pose2d import MediapipePose
from data import generate_and_save_config, generate_config, get_calibration_result, to_dict
from capture import MotionCapture, Status
from utils import TimeUtil


m_capture = MotionCapture()


class Request:
    def __init__(self, string):
        try:
            request = json.loads(string)
            self.id = request['id']
            self.type = request['type']
            self.action = request['action']
            self.data = request.get('data', {})
        except Exception as e:
            raise ValueError(f"Invalid request format: {str(e)}")


class MotionCaptureWebSocket(WebSocket):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_threads = {}

    def handleConnected(self):
        print(self.address[0], 'connected')

    def handleClose(self):
        print(self.address[0], 'closed')

    def handleMessage(self):
        """Handles incoming messages from the client."""
        request = Request(self.data)
        print(f"received: {request.type}-{request.action}")

        if request.type == "test":
            self.send_response(request, "success", "received: test")

        elif request.type == "time":
            unixtime = int(request.data)
            offset = TimeUtil.set_unixtime(unixtime)
            self.send_response(request, "success", f"offset={int(offset)}msec")

        elif request.type == "config":
            data = json.loads(request.data)
            config_path = data['config_path']
            config = generate_config(data)  
            with open(config_path, 'w') as f:
                json.dump(config, f, inednt=4)
            success, res = m_capture.load_config(config_path)
            if success:
                self.send_response(request, "success", res)
            else:
                self.send_response(request, "failed", res)

        elif request.type == "calibration":
            thread = threading.Thread(target=handle_calibration, args=(self, request))
            thread.start()
            self.active_threads[request.type] = thread

        elif request.type == "capture":
            thread = threading.Thread(target=handle_capture, args=(self, request))
            thread.start()
            self.active_threads[request.type] = thread

    def send_response(self, request:Request, status, message=None, data=None):
        """Sends a response back to the client."""
        response = {
            "id" : request.id,
            "type": request.type,
            "action": request.action,
            "status": status
        }
        if message:
            response["message"] = message
        if data:
            response["data"] = json.dumps(data)
        self.sendMessage(json.dumps(response))

    def send_pose(self, data):
        """Sends pose data to the client."""
        response = {
            "type": "pose",
            "data": json.dumps(data)
        }
        self.sendMessage(json.dumps(response))


def handle_capture(ws: MotionCaptureWebSocket, request: Request):
    """Handles capture requests."""
    try:
        if request.action == 'init':
            config_path = request.data if request.data else "config.json"
            success, res = m_capture.load_config(config_path)
            if success:
                ws.send_response(request, 'success', res)
            else:
                ws.send_response(request, 'failed', res)     

        elif request.action == 'start':
            success, res = m_capture.start()
            if success:
                ws.send_response(request, 'success', res)

                # Start a new thread to send data
                sender_thread = threading.Thread(target=data_sender, args=(ws,))
                sender_thread.start()
            else:
                ws.send_response(request, 'failed', res)            

        elif request.action == 'end':
            success, res = m_capture.end()
            if success:
                ws.send_response(request, 'success', res)
            else:
                ws.send_response(request, 'failed', res)

    except Exception as e:
        ws.send_response(request, "failed", str(e))

    finally:
        # Cleanup: Remove the completed thread from active_threads
        if request.type in ws.active_threads:
            del ws.active_threads[request.type]


def handle_calibration(ws: MotionCaptureWebSocket, request: Request):
    """Handles calibration requests."""
    try:            
        if request.action == 'init':
            # Initialize calibration
            success, res = m_capture.init_calibration()
            if success:
                ws.send_response(request, "initialized", res)
            else:
                ws.send_response(request, "failed", res)
                return

        elif request.action == 'start':
            reference_height = float(request.data) if request.data else 1.6  # Default to 1.6 if height is not provided

            # Start calibration
            success, res = m_capture.start_calibration(reference_height=reference_height)

            if success:
                calibration_result = get_calibration_result(m_capture.config)
                ws.send_response(request, "finished", res+f" with height:{reference_height}", data=calibration_result)
            else:
                ws.send_response(request, "failed", res)   
            
        elif request.action == "trajectry":
            trajectory_data = json.loads(request.data)
            success, res = m_capture.correct_calibration_with_hmd_trajectory(trajectory_data)
            if success:
                calibration_result = get_calibration_result(m_capture.config)
                ws.send_response(request, "success", res, data=calibration_result)
            else:
                ws.send_response(request, "failed", res)
                return

    except Exception as e:
        ws.send_response(request, "failed", str(e))

    finally:
        # Cleanup: Remove the completed thread from active_threads
        if request.type in ws.active_threads:
            del ws.active_threads[request.type]


def data_sender(ws: MotionCaptureWebSocket):
    """Send data fetched from the motion capture to the client."""
    while m_capture.is_playing:
        timestamp, keypoints2d_list, keypoints3d = m_capture.read()
        if timestamp:
            data = to_dict(timestamp, keypoints3d, MediapipePose.KEYPOINT_DICT, MediapipePose.Type)
            ws.send_pose(data)
        time.sleep(0.01)


if __name__ == '__main__':
    server = SimpleWebSocketServer('', 50000, MotionCaptureWebSocket)
    server.serveforever()
