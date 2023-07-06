import time

class TimeUtil:

    # return unix milliseconds
    @staticmethod
    def get_unixtime():
        return int(time.time()*1000)


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