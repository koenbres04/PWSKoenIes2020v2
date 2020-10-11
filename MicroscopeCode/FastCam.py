import time
import picamera
import numpy as np
from PIL import Image
from FindTheDot import *
import threading

# record some images over a short period of time
def record(duration, size, rounded_size, framerate, sensor_mode=4, do_print=True):
    frames = []
    class MyOutput(object):
    
        def write(self, buf):
            # write will be called once for each frame of output. buf is a bytes
            # object containing the frame data in YUV420 format; we can construct a
            # numpy array on top of the Y plane of this data quite easily:
            y_data = np.frombuffer(
                buf, dtype=np.uint8, count=rounded_size[0]*rounded_size[1]*3).reshape((rounded_size[1], rounded_size[0], 3))
            # do whatever you want with the frame data here... I'm just going to
            # print the maximum pixel brightness:
    
            frames.append(y_data[:size[1], :size[0], :])
    
        def flush(self):
            # this will be called at the end of the recording; do whatever you want
            # here
            pass
    
    with picamera.PiCamera(
            sensor_mode=sensor_mode,
            resolution='{}x{}'.format(str(size[0]), str(size[1])),
            framerate=framerate) as camera:
        time.sleep(2) # let the camera warm up and set gain/white balance
        output = MyOutput()
        if do_print:
            print("recording...")
        camera.start_recording(output, 'rgb')
        camera.wait_recording(duration) # record 10 seconds worth of data
        camera.stop_recording()
        if do_print:
            print("done!")
        return frames

class ContinuedRecordOutput:

    def __init__(self, owner):
        self.owner = owner
    
    def write(self, buf):
        # write will be called once for each frame of output. buf is a bytes
        # object containing the frame data in YUV420 format; we can construct a
        # numpy array on top of the Y plane of this data quite easily:
        y_data = np.frombuffer(
            buf, dtype=np.uint8, count=self.owner.rounded_size[0]*self.owner.rounded_size[1]*3).reshape((self.owner.rounded_size[1], self.owner.rounded_size[0], 3))
        # do whatever you want with the frame data here... I'm just going to

        self.owner.on_record(y_data[:self.owner.size[1], :self.owner.size[0], :])
    
    def flush(self):
        # this will be called at the end of the recording; do whatever you want
        # here
        pass


# when inheriting this class, create the functions 'do_continue' and 'on_record'

class ContinuedRecord(threading.Thread):
    def initialize(self, size, rounded_size, framerate, sensor_mode=4,
                   macro_frame_duration=0.1):
        self.size = size
        self.rounded_size = rounded_size
        self.framerate = framerate
        self.sensor_mode = sensor_mode
        self.is_paused = False
        self.last_is_paused = False
        self.macro_frame_duration = macro_frame_duration

    def run(self):
        with picamera.PiCamera(
                sensor_mode=self.sensor_mode,
                resolution='{}x{}'.format(str(self.size[0]), str(self.size[1])),
                framerate=self.framerate) as camera:
            time.sleep(2) # let the camera warm up and set gain/white balance
            output = ContinuedRecordOutput(self)
            camera.start_recording(output, 'rgb')
            while self.do_continue():
                # stop or start recording when self.is_paused changes
                if self.is_paused and not self.last_is_paused:
                    camera.stop_recording()
                elif not self.is_paused and self.last_is_paused:
                    camera.start_recording(output, 'rgb')
                    
                self.last_is_paused = self.is_paused
                
                if self.is_paused:
                    self.on_pause()
                else:
                    # record macro_frame_duration seconds worth of data:
                    camera.wait_recording(self.macro_frame_duration) 
            if not self.is_paused:
                camera.stop_recording()


