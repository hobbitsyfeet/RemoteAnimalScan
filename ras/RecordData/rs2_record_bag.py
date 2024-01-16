import pyrealsense2 as rs
import time
from datetime import date, datetime, timedelta

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

now = datetime.now()
date_time = now.strftime("%m_%d_%Y_%H_%M_%S_%f")
file_name = date_time + 'bag'
config.enable_record_to_file(file_name)


pipeline.start(config)

try:
    start=time.time()
    while time.time() - start < 2:
        pipeline.wait_for_frames()
finally:
    pipeline.stop()