# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import time

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
parser.add_argument("-o", "--output", type=str, help="Path to output folder")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given

if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

file_name = os.path.splitext(args.input)[0].replace("\\", "/").split('/')[-1]
queue = rs.frame_queue(capacity=100, keep_frames=True)
try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input, repeat_playback=False)

    # Configure the pipeline to stream the depth stream
    # Change this parameters according to the recorded bag file resolution
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)


    # Create opencv window to render image in
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    
    # Create colorizer object
    colorizer = rs.colorizer()
    queue = rs.frame_queue(100, keep_frames=True)

    pipeline.start(config)
    # Streaming loop
    Frame = True
    while Frame:
        # get frame from pipeline
        try:
            frame = pipeline.wait_for_frames()
            frame_number = frame.frame_number

            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()
            depth_color_image = np.asanyarray(depth_frame.get_data())
            colour_image = np.asanyarray(color_frame.get_data())
            print(frame_number)
            cv2.imshow("Colour Stream", colour_image)
            cv2.imshow("Depth Stream", depth_color_image)
            frame.keep()
            queue.enqueue(frame)
        except:
            print("Reached EOF")
            break
        
    pipeline.stop() 

    print("Slower callback + keeping queue")
    
    pipeline.start(config, queue)
    counter = 0
    while True:
        frames = queue.wait_for_frame()
        color_frame = frames.as_frameset().get_color_frame()
        depth_frame = frames.as_frameset().get_depth_frame()

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        colour_image = np.asanyarray(color_frame.get_data())

        if args.output:
            
            cloud = rs.pointcloud()
            cloud.map_to(color_frame)
            points = cloud.calculate(depth_frame)
            os.path.splitext(args.input)[0]


            save_str = args.output + file_name + "-" + str(counter) + ".ply"
            print("exporting ", save_str)
            points.export_to_ply(save_str, color_frame)
        # Render image in opencv window
        cv2.imshow("Colour Stream", colour_image)
        cv2.imshow("Depth Stream", depth_color_image)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        counter += 1
    pipeline.stop()


    

            

    

finally:
    pass
