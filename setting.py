'''
カメラパラメータの設定用サーバ
'''

import cv2
from flask import *
import subprocess

import numpy as np

PYTHON_PATH = r"/path/to/python.exe"
SCRIPT_PATH = r"/path/to/project/run.py"

global process
process = None

app = Flask(__name__)

@app.route("/", methods=["GET"])
def root():
    return ""


# 接続テスト
@app.route("/test", methods=["GET"])
def test():
    return "Connection Established"


# カメラ設定
@app.route("/settings", methods=["POST"])
def settings():
    if 'application/json' not in request.headers['Content-Type']:
        return jsonify(res='error'), 400
    try:
        data = request.json
        config_path = save_config(data)
        return f"config saved to {config_path}"
    except:
        return "Configuration Failed"
    

@app.route("/start", methods=["POST"])
def start():
    global process
    if process is not None:
        return 'Capture has already started'
    if 'application/json' not in request.headers['Content-Type']:
        return jsonify(res='error'), 400
    try:
        data = request.json
        config_path = data['config_path']
        udp_port = data['udp_port']
        process = process_start(SCRIPT_PATH, config_path, udp_port)
        return f"Capture started at Port:{udp_port}"
    except Exception as e:
        return f"Failed to start : {e}"


@app.route("/end", methods=["GET"])
def end():
    global process
    try:
        process.terminate()
        process = None
        return "Process Ended"
    except:
        return "Process has not been started"  
    

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


def process_start(script_path, config_path, udp_port):
    command = [PYTHON_PATH, script_path, "--config", config_path, "--port", str(udp_port)]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


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