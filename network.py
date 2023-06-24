'''
For UDP Communication
'''


import socket
import json


class UDPClient:
    def __init__(self, host='127.0.0.1', port=50000):
        self.host = host
        self.port = port
        self.isOpened = False
        self.client = None

    def open(self):
        if self.isOpened:
            self.close()
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.isOpened = True

    def send(self, data):
        if not self.isOpened:
            return False
        try:
            data_json = json.dumps(data)
            self.client.sendto(data_json.encode('utf-8'), (self.host, self.port))
        except Exception as e:
            print(f"Error during sending data: {str(e)}")
            return False
        return True

    def close(self):
        if self.client is not None:
            self.client.close()
        self.isOpened = False


class UDPServer:
    def __init__(self, host='127.0.0.1', port=50000, timeout=1.0, buffersize=4096):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.buffersize = buffersize
        self.sock = None

    def open(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(self.timeout)
        self.sock.bind((self.host, self.port))
        print('Server started...')

    def listen(self):
        data = None
        if self.sock:
            try:
                data, addr = self.sock.recvfrom(self.buffersize)
            except socket.timeout:
                pass
            except Exception as e:
                print(f"Error during listening: {str(e)}")
        return data
    
    def close(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = None
        print("Server stopped.")



