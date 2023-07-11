'''
カメラパラメータの設定用サーバ
'''

from flask import *
import subprocess

import numpy as np

from controller import MotionCaptureController

controller = MotionCaptureController()

app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    return ""


# 接続テスト
@app.route("/test", methods=["GET"])
def test():
    return "Connection Established", 200


# カメラ設定
@app.route("/settings", methods=["POST"])
def settings():
    if 'application/json' not in request.headers['Content-Type']:
        return jsonify(res='error'), 400
    try:
        data = request.json
        config_path = save_config(data)
        return f"Config saved to {config_path}", 200
    except:
        return "Configuration failed", 202
    

@app.route("/start", methods=["POST"])
def start():
    if 'application/json' not in request.headers['Content-Type']:
        return jsonify(res='error'), 400
    if controller.is_playing:
        return f"Capture has already started", 202
    try:
        data = request.json
        config_path = data['config_path']
        udp_port = data['udp_port']
        controller.initialize(config_path=config_path, udp_host='127.0.0.1', udp_port=udp_port)
        controller.start_capture()
        return f"Capture started at Port:{udp_port}", 200
    except Exception as e:
        return f"Failed to start : {e}", 400
    

@app.route("/end", methods=["GET"])
def end():
    if not controller.is_playing:
        return "Capture has not started yet", 202
    controller.end_capture()
    return "Capture ended", 200
    

def save_config(data):
    config_path = data['config_path']
    config = {'cameras': []}

    for camera in data['cameras']:
        camera_config = {
            'name': camera['name'],
            'setting_path': camera['setting_path']
        }
        if camera['type'] == 'USBCamera':
            camera_config['device'] = camera['device_id']
        elif camera['type'] == 'Video':
            camera_config['device'] = camera['video_path']
        else:
            pass        
        config['cameras'].append(camera_config)
        
        # write extrinsic
        position = camera['position']
        t = np.array([position['x'], position['y'], position['z']]).reshape([3,1])

        rotation = camera['rotation']
        quaternion = [rotation['x'], rotation['y'], rotation['z'], rotation['w']]
        R = quaternion_to_matrix(quaternion)

        Rc = R.T
        tc = -R.T @ t
        extrinsic = np.concatenate([Rc, tc], axis=1)

        try:
            setting_path = camera['setting_path']
            with open(setting_path, 'r') as f:
                camera_setting = json.load(f)
        except:
            camera_setting = {}

        camera_setting['extrinsic_matrix'] = extrinsic.tolist()
        
        with open(setting_path, 'w') as f:
            json.dump(camera_setting, f, indent=4)


    with open(config_path, 'w') as fp:
        json.dump(config, fp, indent=4)

    print(f"config saved at {config_path}")
    return config_path


def quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion
    matrix = np.zeros([3,3])
    matrix[0,0] = 2*w**2 + 2*x**2 - 1
    matrix[0,1] = 2*x*y - 2*z*w
    matrix[0,2] = 2*x*z + 2*y*w
    matrix[1,0] = 2*x*y + 2*z*w
    matrix[1,1] = 2*w**2 + 2*y**2 - 1
    matrix[1,2] = 2*y*z - 2*x*w
    matrix[2,0] = 2*x*z - 2*y*w
    matrix[2,1] = 2*y*z + 2*x*w
    matrix[2,2] = 2*w**2 + 2*z**2 - 1
    return matrix


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8888)