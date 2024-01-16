from argparse import ArgumentParser
from logging import exception

import cv2

# from helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback
from pyk4a import PyK4ACapture
from pyk4a import PyK4A
import pyk4a




def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def play(playback: PyK4APlayback):
    while True:
        try:
            print(playback.configuration)
            # playback.calibration()
            capture: PyK4ACapture = playback.get_next_capture()
            
            # print(capture.col)
            capture.depth
            if capture.color is not None:
                cv2.imshow("Color", capture.color)
                try:
                    print(capture.color_exposure_usec)
                except Exception as e:
                    print(e)
                
                try:
                    print(capture.color_iso_speed)
                except Exception as e:
                    print(e)
                    
            if capture.depth is not None:
                cv2.imshow("Depth", capture.depth)
            if capture.ir is not None:
                cv2.imshow("IR", capture.ir)
                if capture.ir.any() and capture.depth.any():
                    cv2.imshow("IR/Depth", capture.ir/capture.depth)
                
        #                 cap = self.camera.get_capture()
        # # points = cap.depth_point_cloud.reshape((-1, 3))
        # # colors = cap.transformed_color[..., (2, 1, 0)].reshape((-1, 3))
        # # print(colors.tolist())
        # # pointcloud = pyk4a.transformation.depth_image_to_point_cloud(cap.depth, cap._calibration,cap._calibration.thread_safe, calibration_type_depth=True)
        # # print(pointcloud.tolist())
        # # print(cap.color)
        
        # # color = pyk4a.transformation.color_image_to_depth_camera( cap.color, cap.depth,cap._calibration, cap._calibration.thread_safe)

        # # cap._calibration.color_to_depth_3d()
        # # print(color.shape)
        # # print(color.tolist())
        # # colors = cap.transformed_color
        # # print(colors.tolist())

        # # print(points.shape)
        # # print(colors.shape)

        # # o3d_cloud = numpy_to_o3d(points, colors)
        # # o3d.visualization.draw_geometries_with_editing([o3d_cloud])

            key = cv2.waitKey(10)
            if key != -1:
                break
        except EOFError:
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = ArgumentParser(description="pyk4a player")
    parser.add_argument("--seek", type=float, help="Seek file to specified offset in seconds", default=0.0)
    parser.add_argument("FILE", type=str, help="Path to MKV file written by k4arecorder")

    args = parser.parse_args()
    filename: str = args.FILE
    offset: float = args.seek

    playback = PyK4APlayback(filename)
    playback.open()

    info(playback)

    if offset != 0.0:
        playback.seek(int(offset * 1000000))
    play(playback)

    playback.close()


if __name__ == "__main__":
    filename = "K:/Data/NightScan/25_07_2022-22_52_21.mkv"
    offset: float = 0

    playback = PyK4APlayback(filename)
    
    playback.open()

    info(playback)

    if offset != 0.0:
        playback.seek(int(offset * 10000))
    play(playback)

    playback.close()