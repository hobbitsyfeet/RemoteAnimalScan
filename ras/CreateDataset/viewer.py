import sys
from PyQt5.QtWidgets import (QWidget,
                         QPushButton, QApplication, QGridLayout)
from PyQt5.QtCore import QThread, QObject, pyqtSignal

import open3d as o3d
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class Action_Poll(QObject):
    finished = pyqtSignal()  # give worker class a finished signal
    def __init__(self, viewer=None, parent=None):
        # self.viewer = viewer
        self.parent = parent
        
        QObject.__init__(self, parent=parent)
        self.continue_run = True  # provide a bool run condition for the class
    

    def poll_actions(self, visualizer):
        

        ctrl = visualizer.get_view_control()
        camera_params = ctrl.convert_to_pinhole_camera_parameters()
        
        # rotate the camera 180 degrees around the x axis
        rot = np.eye(4)
        rot[:3, :3] = R.from_euler('x', 180, degrees=True).as_dcm()
        rot = rot.dot(camera_params.extrinsic)

        # set z=1 since data exists in negative depth
        rot[2,3] = 1000

        # Apply transformations to camera
        camera_params.extrinsic = rot
        ctrl.convert_from_pinhole_camera_parameters(camera_params)

        # Update the visualizer with the camera transformation
        visualizer.update_renderer()

        while self.continue_run:  # give the loop a stoppable condition
            try:
                if not visualizer.poll_events():
                    print("Closing")
                    self.continue_run = False
                    break
                else:
                    camera_params = ctrl.convert_to_pinhole_camera_parameters()
                    # points = visualizer.get_picked_points()
                    # print(points.coord)

                    # if len(points) >= 1:
                    #     # if self.parent.viewer.current_polygon is None:
                    #         # self.parent.add_label()

                    #     if self.current_polygon is not None:
                    #             print("Left Button Clicked")
                    #             # self.current_polygon.assign_point(mapped_point)
                    #             self.redraw_image()
                    #             image = self.current_polygon.draw(self.image)
                    #             self.update_image(image)
                    #             # self.update_distance(mapped_point)
                    # print(camera_params.extrinsic)
                    # print(camera_params.extrinsic[2,3])
                    visualizer.update_renderer()
                    
                    # self.parent.o3d_vis.register_key_callback(32,self.stop)
            except Exception as e:
                print(e)
                break

        visualizer.close()
        visualizer.destroy_window()
        del self.parent.o3d_vis
        self.parent.o3d_vis = None
        self.finished.emit() 

        

    def stop(self):
        self.continue_run = False  # set the run condition to false on stop
 