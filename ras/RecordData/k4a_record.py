import subprocess
from ffmpeg_writer import ffmpeg_writer
from k4a_camera import k4a_camera

import time
import threading

# https://docs.microsoft.com/en-us/azure/kinect-dk/hardware-specification
# RGB Camera Resolution (HxV) 	Aspect Ratio 	Format Options 	Frame Rates (FPS) 	Nominal FOV (HxV)(post-processed)
# 3840x2160 	                16:9 	        MJPEG 	        0, 5, 15, 30 	    90°x59°
# 2560x1440 	                16:9 	        MJPEG 	        0, 5, 15, 30 	    90°x59°
# 1920x1080 	                16:9 	        MJPEG 	        0, 5, 15, 30 	    90°x59°
# 1280x720 	                    16:9 	        MJPEG/YUY2/NV12 0, 5, 15, 30 	    90°x59°
# 4096x3072 	                4:3 	        MJPEG 	        0, 5, 15 	        90°x74.3°
# 2048x1536 	                4:3 	        MJPEG 	        0, 5, 15, 30 	    90°x74.3°

# Mode 	                Resolution 	FoI 	    FPS 	Operating range* 	Exposure time
# NFOV unbinned 	    640x576 	75°x65° 	0, 5, 15, 30 	0.5 - 3.86 m 	12.8 ms
# NFOV 2x2 binned(SW) 	320x288 	75°x65° 	0, 5, 15, 30 	0.5 - 5.46 m 	12.8 ms
# WFOV 2x2 binned 	    512x512 	120°x120° 	0, 5, 15, 30 	0.25 - 2.88 m 	12.8 ms
# WFOV unbinned 	    1024x1024 	120°x120° 	0, 5, 15 	    0.25 - 2.21 m 	20.3 ms
# Passive IR 	        1024x1024 	N/A 	    0, 5, 15, 30 	N/A 	1.6 ms

NFOV_UNBINNED = {"Resolution": (640,576), "FoI_Deg":(75,65), "FPS": (0,5,15,30), "Range_meters":(0.5, ) } #
WFOV_UNBINNED = "WFOV_UNBINNED" #

NFOV_2X2BINNED = "NFOV_2X2BINNED"
WFOV_2X2BINNED = "WFOV_2X2BINNED"

KEY_RESOLUTION = "Resolution"
KEY_FOI = "FoI_Deg"
KEY_FPS = "FPS"
KEY_RANGE = "Range_meters"

FPS_30 = "30"
FPS_15 = "15"
FPS_5 = "5"

RES_720P = "720p"
RES_1080P = "1080p"
RES_1440P = "1440p"
RES_2160P = "2160p"
RES_3072P = "3072p"

RES_720_TUPLE = (1280,720)
class k4a_recorder():
    def __init__(self, k4a_record_exe="ffmpeg", res_colour = (1920,1080), fov = NFOV_UNBINNED, fps=30):
        # self.record_exe = "C:/Program Files/Azure Kinect SDK v1.4.1/tools/k4arecorder.exe"
        self.fps = fps
        self.camera = k4a_camera()
        self.recorder = ffmpeg_writer("Test", res_colour , fov[KEY_RESOLUTION], fps=fps)
        self.running = True
        self.color_depth_thread = threading.Thread(target=self.start)
        # self.end_thread = threading.Thread(target=self.start)
        self.read_thread = threading.Thread(target=self.get_input)
        

    def start_record(self):
        # video_thread = threading.Thread(target=self.record)
        print("RECORDING")
        self.color_depth_thread.start()
        self.read_thread.start()

    def start(self):
        self.recorder.write_colour_start()
        self.recorder.write_depth_start()
        # self.recorder.write_depth_start()
        # self.running = True
        # self.recorder.write_colour()
        while self.running:
            # print(self.running)
            colour, depth = self.camera.read()
            # print(self.camera.calibraion.)
            # print(type(colour))
            # print(colour[...,:3].shape)
            # print(colour[...,:3])
            # print(color.dtype())
            # print(colour)
            # self.recorder.write_colour(colour[...,:3])
            # self.recorder.write_depth(depth)
            # self.recorder.write_depth(depth)
            # time.sleep(1/self.fps)
   
    def get_input(self):
        test = input("Enter Q to quit")
        if test == 'q' or test == 'Q':
            self.running = False
            

    def end(self):
        print("CLOSING")
        self.color_depth_thread.join()
        self.read_thread.join()
        

        self.recorder.close()
        self.running = False
        # self.color_depth_thread.join()q


if __name__ == "__main__":
    recorder = k4a_recorder()
    print("Starting")
    recorder.start_record()
    time.sleep(5)
    recorder.end()
    print("Stopping")