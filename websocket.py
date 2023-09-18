import json
import os
import threading
import time
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
from capture import MotionCapture, Status


m_capture = MotionCapture()


class Request:
    def __init__(self, string):
        try:
            request = json.loads(string)
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
            self.send_response(request.type, request.action, "success")

        elif request.type == "calibration":
            thread = threading.Thread(target=handle_calibration, args=(self, request))
            thread.start()
            self.active_threads[request.type] = thread

        elif request.type == "capture":
            thread = threading.Thread(target=handle_capture, args=(self, request))
            thread.start()
            self.active_threads[request.type] = thread

    def send_response(self, type, action, status, message=None, data=None):
        """Sends a response back to the client."""
        response = {
            "type": type,
            "action": action,
            "status": status
        }
        if message:
            response["message"] = message
        if data:
            response["data"] = data
        self.sendMessage(json.dumps(response))

    def send_pose(self, data):
        """Sends pose data to the client."""
        response = {
            "type": "pose",
            "data": data
        }
        self.sendMessage(json.dumps(response))


def handle_capture(ws: MotionCaptureWebSocket, request: Request):
    """Handles capture requests."""
    try:
        if request.action == 'start':
            success, res = m_capture.start()
            if success:
                ws.send_response(request.type, request.action, 'success', res)

                # Start a new thread to send data
                sender_thread = threading.Thread(target=data_sender, args=(ws,))
                sender_thread.start()
            else:
                ws.send_response(request.type, request.action, 'failed', res)            

        elif request.action == 'end':
            success, res = m_capture.end()
            if success:
                ws.send_response(request.type, request.action, 'success', res)
            else:
                ws.send_response(request.type, request.action, 'failed', res)

    except Exception as e:
        ws.send_response(request.type, request.action, "failed", str(e))

    finally:
        # Cleanup: Remove the completed thread from active_threads
        if request.type in ws.active_threads:
            del ws.active_threads[request.type]


def handle_calibration(ws: MotionCaptureWebSocket, request: Request):
    """Handles calibration requests."""
    output_dir = './data'
    try:
        if request.action == 'start':
            reference_height = request.data.get('height', 1.6)  # Default to 1.6 if height is not provided
            ws.send_response(request.type, request.action, "initialized")

            # Start calibration
            success, res = m_capture.calibrate(reference_height=reference_height)
            if success:
                ws.send_response(request.type, request.action, "finished", res)
            else:
                ws.send_response(request.type, request.action, "failed", res)
                return

        elif request.action == "trajectry":
            trajectory_data = request.data
            success, res = m_capture.correct_calibration_with_hmd_trajectory(trajectory_data)
            if success:
                ws.send_response(request.type, request.action, "success", res)
            else:
                ws.send_response(request.type, request.action, "failed", res)
                return

    except Exception as e:
        ws.send_response(request.type, request.action, "failed", str(e))

    finally:
        # Cleanup: Remove the completed thread from active_threads
        if request.type in ws.active_threads:
            del ws.active_threads[request.type]


def data_sender(ws: MotionCaptureWebSocket):
    """Send data fetched from the motion capture to the client."""
    while m_capture.is_playing:
        data = m_capture.read()
        if data:
            ws.send_pose(data)
        time.sleep(0.01)



if __name__ == '__main__':
    server = SimpleWebSocketServer('', 50000, MotionCaptureWebSocket)
    server.serveforever()
