import numpy as np
from .camera import CameraSetting


def to_dict(timestamp, keypoints3d, keys, type):
    data = {
        "Type":type, 
        "TimeStamp": timestamp, 
        "Bones":[]
        }

    if keypoints3d is None:
        return data 
    
    for key in keys:
        bone = {
            "Name": key,
            "Position":{
                "x": float(keypoints3d[keys[key],0]),
                "y": float(keypoints3d[keys[key],1]),
                "z": float(keypoints3d[keys[key],2]),
            }
        }
        data['Bones'].append(bone)  
    return data


def generate_config(data):
    config = {'cameras': []}

    for camera in data['cameras']:
        # 設定ファイル名
        setting_path = camera['setting_path']
        camera_config = {
            'name': camera['name'],
            'setting_path': setting_path,
        }
        if camera['type'] == 'USBCamera':
            camera_config['device'] = camera['device_id']
        elif camera['type'] == 'Video':
            camera_config['device'] = camera['video_path']
        else:
            camera_config['device'] = ""            
        config['cameras'].append(camera_config)

        # write extrinsic
        position = camera['position']
        t = np.array([position['x'], position['y'], position['z']]).reshape([3,1])

        rotation = camera['rotation']
        quaternion = [rotation['x'], rotation['y'], rotation['z'], rotation['w']]
        R = quaternion_to_matrix(quaternion)

        Rc = R.T
        tc = -R.T @ t
        extrinsic = np.concatenate([Rc, tc], axis=-1)

        # save extrinsic
        camera_setting = CameraSetting(setting_path)
        camera_setting.extrinsic_matrix = extrinsic
        camera_setting.save()

    return config


def get_calibration_result(config):
    data = {"cameras":[]}
    for camera in config['cameras']:
        camera_setting = CameraSetting(camera['setting_path'])
        Rc = camera_setting.extrinsic_matrix[:,:3]
        tc = camera_setting.extrinsic_matrix[:,3:]
        R = Rc.T
        t = -Rc.T @ tc
        pos_x, pos_y, pos_z = t.flatten()
        rot_x, rot_y, rot_z, rot_w = matrix_to_quaternion(R)
        camera_config = {
            "name": camera['name'],
            "position": {
                "x": -pos_x,
                "y": pos_y,
                "z": pos_z
            },
            "rotation":{
                "x": rot_x,
                "y": -rot_y,
                "z": -rot_z,
                "w": rot_w
            }
        }
        data['cameras'].append(camera_config)
    return data


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

def matrix_to_quaternion(matrix):
    # Ensure the matrix is a numpy array
    matrix = np.asarray(matrix)
    
    # Calculate the trace of the matrix
    tr = matrix.trace()
    
    # Check the trace value to perform the suitable calculation
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (matrix[2, 1] - matrix[1, 2]) / S
        y = (matrix[0, 2] - matrix[2, 0]) / S
        z = (matrix[1, 0] - matrix[0, 1]) / S
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / S
        x = 0.25 * S
        y = (matrix[0, 1] + matrix[1, 0]) / S
        z = (matrix[0, 2] + matrix[2, 0]) / S
    elif matrix[1, 1] > matrix[2, 2]:
        S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / S
        x = (matrix[0, 1] + matrix[1, 0]) / S
        y = 0.25 * S
        z = (matrix[1, 2] + matrix[2, 1]) / S
    else:
        S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / S
        x = (matrix[0, 2] + matrix[2, 0]) / S
        y = (matrix[1, 2] + matrix[2, 1]) / S
        z = 0.25 * S
        
    return x, y, z, w