import time

class TimeUtil:

    # return unix milliseconds
    @staticmethod
    def get_unixtime():
        return int(time.time()*1000)