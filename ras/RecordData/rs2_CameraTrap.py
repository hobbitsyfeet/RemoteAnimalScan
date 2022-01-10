"""
Camera Trap is a module which allows the use of depth cameras to be triggered automatically when something is in range.
    Requirements: Depth Camera (Intel Realsense, Kinect Azure)
    Optional: IR motion sensor

"""
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import date, datetime, timedelta

def record(output_path, width = 1280, height=720, timer=7):

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

    # Start streaming
    pipeline.start(config)

    cloud = rs.pointcloud()

    #Set recording time
    time_start = datetime.now()
    time_end = time_start + timedelta(seconds=timer)

    while datetime.now() < time_end:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        

        if frames:
            depth_frame = frames.get_depth_frame()
            
            points = cloud.calculate(depth_frame)
            cloud.map_to(color_frame)
            now = datetime.now()
            date_time = now.strftime("%m_%d_%Y_%H_%M_%S_%f")

            save_str = (save_path + date_time + ".ply")
            points.export_to_ply(save_str, color_frame)

        # Display camera
        # color_image = np.asanyarray(color_frame.get_data())
        # cv2.imshow('Frame', color_image)
        # cv2.waitKey(1)


    # Stop streaming
    pipeline.stop()

if __name__ == "__main__":
    save_path = "K:/Github/RemoteAnimalScan/data/"
    record(save_path)

    
