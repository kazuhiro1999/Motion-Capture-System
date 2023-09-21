import time

class TimeUtil:
    Offset = 0

    # return unix milliseconds
    @staticmethod
    def get_unixtime():
        current_time = time.time() 
        return int(current_time*1000)
    
    # set unix milliseconds
    @staticmethod
    def set_unixtime(ut):
        offset = ut - TimeUtil.get_unixtime()
        TimeUtil.Offset = offset
        return offset


import time

class FPSCalculator:
    DICT = {}

    @staticmethod
    def start(name):
        FPSCalculator.DICT.setdefault(name, {'start':[], 'end':[], 'time':[]})
        if len(FPSCalculator.DICT[name]['start']) == len(FPSCalculator.DICT[name]['end']):
            t = time.time()
            FPSCalculator.DICT[name]['start'].append(t)
        else:
            print(f"{name} has already started")

    @staticmethod
    def end(name):        
        if name not in FPSCalculator.DICT.keys():
            return 
        if len(FPSCalculator.DICT[name]['end']) == len(FPSCalculator.DICT[name]['start']) - 1:
            t = time.time()
            FPSCalculator.DICT[name]['end'].append(t)
            elapsed_time = t - FPSCalculator.DICT[name]['start'][-1]
            FPSCalculator.DICT[name]['time'].append(elapsed_time)
        else:
            print(f"{name} has not started")

    @staticmethod
    def get_execution_time(name, duration=1):
        if name not in FPSCalculator.DICT.keys():
            return 0
        count = len(FPSCalculator.DICT[name]['time'])
        if count == 0:
            return 0
        elif count >= duration:
            return sum(FPSCalculator.DICT[name]['time'][-duration:]) / duration
        else:
            return sum(FPSCalculator.DICT[name]['time']) / count

    @ staticmethod
    def calc_fps(func):
        def wrapper(*args, **kwargs):
            FPSCalculator.start(func.__name__)
            res = func(*args, **kwargs)
            FPSCalculator.end(func.__name__)
            return res
        return wrapper


import subprocess
import re

def get_camera_devices():

    command = "ffmpeg -list_devices true -f dshow -i dummy"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    video_devices = []

    device_name = ""
    for line in iter(process.stdout.readline, b''):
        line = line.decode('utf-8').strip()
        if '(video)' in line:  # デバイスがビデオデバイスかどうかを確認
            match = re.search(r'".*"', line)  # デバイス名はダブルクォーテーションで囲まれている
            if match:
                device_name = match.group().strip('"')  # ダブルクォーテーションを削除
        elif "Alternative name" in line and device_name: # Alternative nameの行かつ、device_nameが既に取得済みの場合
            match = re.search(r'".*"', line)  # Alternative nameはダブルクォーテーションで囲まれている
            if match:
                alternative_name = match.group().strip('"')  # ダブルクォーテーションを削除
                video_devices.append({"device_name": device_name, "alternative_name": alternative_name})
                device_name = ""  # device_nameをリセット

    return video_devices


if __name__ == '__main__':
    devices = get_camera_devices()
    for device in devices:
        print(device)