import json
import os
import threading
import time
import PySimpleGUI as sg
from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket

from space_calibration import calibration_process


class MotionCaptureWebSocket(WebSocket):

    def handleConnected(self):
        print(self.address[0], 'connected')

    def handleClose(self):
        print(self.address[0], 'closed')

    def handleMessage(self):
        # クライアントからのメッセージを受け取る
        data = json.loads(self.data)
        type = data['type']
        action = data['action']
        print(f"received: {type}-{action}")
        
        if type == "test":
            response = self.send_response(type, action, "success")
        if type == "calibration" and action == "start":
            # キャリブレーションを開始
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(script_dir, data['data']['config_path'])
                height = data['data']['height']                
                self.output_dir = sg.popup_get_folder("保存先を選んでね") # 保存用
                # キャリブレーションを別スレッドで実行
                calibration_thread = threading.Thread(target=calibration_process, args=(config_path, height, self.output_dir))
                calibration_thread.start()
                self.send_response(type, action, "initialized")
                response_message = {
                    "type": "calibration",
                    "action": "start",
                    "status": "finished"
                }
                # 非同期でスレッドの終了を監視する
                self.monitor_thread(calibration_thread, response_message)
            except Exception as e:
                self.send_response(type, action, "failed")
        if type == "calibration" and action == "trajectry":
            trajectry = data['data']
            with open(f"{self.output_dir}/trajectry.json", 'w') as f:
                json.dump(trajectry, f, indent=4)
            self.send_response(type, action, "success")
        # 他のアクションも同様...

    def send_response(self, type, action, status, message=None, data=None):
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

    # スレッドの終了を監視
    def monitor_thread(self, thread, response):
        while thread.is_alive():
            time.sleep(0.5)  # 0.5秒ごとにスレッドの状態を確認

        # スレッドが終了したら指定されたメッセージを送信
        self.sendMessage(json.dumps(response))



server = SimpleWebSocketServer('', 50000, MotionCaptureWebSocket)
server.serveforever()
