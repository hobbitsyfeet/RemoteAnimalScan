from datetime import datetime

import numpy as np
import open3d as o3d
import pyk4a
from pyk4a import ColorResolution, Config, PyK4A, PyK4ACapture
import cv2

NFOV_UNBINNED = pyk4a.DepthMode.NFOV_UNBINNED
WFOV_UNBINNED = pyk4a.DepthMode.WFOV_UNBINNED

NFOV_2X2BINNED = pyk4a.DepthMode.NFOV_2X2BINNED
WFOV_2X2BINNED = pyk4a.DepthMode.WFOV_2X2BINNED


FPS_30 = pyk4a.FPS.FPS_30
FPS_15 = pyk4a.FPS.FPS_15
FPS_5 = pyk4a.FPS.FPS_5

RES_720P = ColorResolution.RES_720P
RES_1080P = ColorResolution.RES_1080P
RES_1440P = ColorResolution.RES_1440P
RES_2160P = ColorResolution.RES_2160P
RES_3072P = ColorResolution.RES_3072P


def numpy_to_o3d(np_cloud_points, np_cloud_colors=None, np_cloud_normals=None, swap_RGB=False, monochrome=False):
    #create o3d pointcloud and assign it
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np_cloud_points)
    if np_cloud_colors is not None:
        if monochrome:
            R = np_cloud_colors[:].copy()
            G = np_cloud_colors[:].copy()
            B = np_cloud_colors[:].copy()
            print(R)
            # np_cloud_colors = np_cloud_colors[:]
            # np_cloud_colors[:] = np.array(R,G,B)
            # np_cloud_colors[:] = R
            # np_cloud_colors[:] = G
            # np_cloud_colors[:] = B

            print(np_cloud_colors)
            
            
        if swap_RGB:
            R = np_cloud_colors[:,0].copy()
            # G = np_cloud_colors[:,1].copy()
            B = np_cloud_colors[:,2].copy()
            
            np_cloud_colors[:,0] = B
            np_cloud_colors[:,2] = R



        o3d_cloud.colors = o3d.utility.Vector3dVector(np_cloud_colors.astype(np.float) / 255.0)
    if np_cloud_normals is not None:
        o3d_cloud.normals = o3d.utility.Vector3dVector(np_cloud_normals)

    return o3d_cloud

class k4a_camera():
    def __init__(self, resolution = ColorResolution.RES_1080P, framerate = FPS_15, fov = NFOV_2X2BINNED):
        """
        Intuitive K4A Camera Class. Initialize this class with parameters and simply read the data.
        """
        

        self.resolution = resolution
        self.framerate = framerate
        self.fov = fov
        
        self.camera = self.setup_k4a()
        self.cap = self.camera.get_capture()

        self.calibraion = pyk4a.calibration.Calibration(self.cap._capture_handle,self.fov,self.resolution)



    def setup_k4a(self):


        k4a = PyK4A(Config(color_resolution=self.resolution,
                        depth_mode=self.fov,
                        camera_fps=self.framerate,
                        synchronized_images_only=True))
        k4a.open()
        k4a.start()
        return k4a


    def read_imu(self):
        imu = self.camera.get_imu_sample()
        return

    def read(self):
        cap = self.camera.get_capture()
        return cap.color, cap.depth

    def stop(self):
        self.camera.stop()
        return 0
    
    def get_intrinsics(self):
        return self.camera.calibration_raw()

    def get_fps(self):
        if self.framerate == FPS_30:
            return 30
        elif self.framerate == FPS_15:
            return 15
        elif self.framerate == FPS_5:
            return 5
    
    def record_ply(self, folder):
        cap = self.camera.get_capture()
        pyk4a.transformation.color_image_to_depth_camera()

    def write_colour_to_depth(self):
        cap = self.camera.get_capture()
        print(cap.color_iso_speed)
        points = cap.depth_point_cloud.reshape((-1, 3))
        # colors = cap.transformed_color[..., (2, 1, 0)].reshape((-1, 3))
        # print(colors.tolist())
        pointcloud = pyk4a.transformation.depth_image_to_point_cloud(cap.depth, cap._calibration,cap._calibration.thread_safe, calibration_type_depth=True)
        # print(pointcloud.tolist())
        # print(cap.color)
        # colour = pyk4a.depth_image_to_color_camera_custom(cap.depth, cap.ir, cap._calibration, thread_safe=True)

        cv2.imshow("Colour", cap.color)
        cv2.waitKey(1)
        # depth = pyk4a.depth_image_to_point_cloud(colour, cap._calibration, thread_safe=True)
        # points = cap.depth_point_cloud.reshape((-1, 3))
        
        # points = cap.depth_point_cloud.reshape((-1, 3))
        # color = pyk4a.transformation.color_image_to_depth_camera( cap.color, cap.depth,cap._calibration, cap._calibration.thread_safe)

        # cap._calibration.color_to_depth_3d()
        # print(color.shape)
        # print(color.tolist())
        # colors = cap.transformed_color[..., (2, 1, 0)].reshape((-1, 3))
        # print(colors.tolist())

        # print(points.shape)
        # print(colors.shape)
        # cap.ir.reshape(-1, 3)
        # print(cap.ir.tolist())
        # o3d_cloud = numpy_to_o3d(points, cap.transformed_ir.reshape(-1,1), monochrome=True)
        # o3d.visualization.draw_geometries_with_editing([o3d_cloud])
        # print(o3d_cloud)

    def write_depth_to_colour(self, folder):
        pass


        # o3d_cloud = numpy_to_o3d(points, colors)
        # print(o3d_cloud)
        # print("LAAAAAAA")
        # o3d.io.write_point_cloud("Scan_" + (folder+date_time+".ply"), o3d_cloud)

        



if __name__ == "__main__":
    camera = k4a_camera()
    # # print(cam.get_intrinsics())
 
    while(True):

    # #     # colour, depth = cam.read()
        camera.write_colour_to_depth()
        # print(colour)
        # print(cam.cap.transformed_depth_point_cloud)

        # print(colour)
        # cam.record_ply("C:/Users/legom/Documents/GitHub/RemoteAnimalScan/data/")
