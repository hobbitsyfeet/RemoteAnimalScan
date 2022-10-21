from labeling.polygon import LabelPolygon
import labeling.utils as utils
import pandas as pd
from copy import deepcopy
import os
import datetime
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                             QComboBox, QDockWidget, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSpinBox, QStackedLayout, QStyle,
                             QTextEdit, QVBoxLayout, QWidget, QMenuBar)

'''
######################################################################
CLASS DATASET


######################################################################
'''
class Dataset():
    def __init__(self):
        # self.labels = []
        # self.polygons = []

        # labels[filename] = (polygons, labels)
        self.file_labels = {}

        self.states = []
        self.previous_states = []

        self.labels = []
        self.current_polygon = None
        self.parent_polygon = None
        self.sibling_polygons = []
        self.current_file = None
        self.image = None
        self.depth = None
        self.point_pairs = None
        self.inverse_pairs = None
        self.cloud = None

        self.redone = False
        self.undone = False

        self.export_path = None



        self.dataframe = pd.DataFrame(columns=['Participant',
                                                'Filename',
                                                'Points',
                                                'Distance',
                                                'TotalDistance',
                                                ])

    # Calls a function to test the number of labels
    def empty(self): 
        return len(self.labels) <= 0

    def undo(self):
        # print("UNDO")
        if self.states:
            self.previous_states.append(self.states[-1])
            polygon = self.states.pop()
            
            print(polygon.points)
            self.labels[polygon.index] = polygon
            self.current_polygon = self.labels[polygon.index]
            # self.previous_states.append(self.current_polygon)
            # self.previous_states.append(self.current_polygon)
            if len(self.previous_states) >= 200:
                self.previous_states.pop(0)

    def redo(self):
        # print("REDO")
        if self.previous_states:
            polygon = self.previous_states.pop()
            print(polygon.points)
            # print(len(self.previous_states))
            self.labels[polygon.index] = polygon
            self.current_polygon = self.labels[polygon.index]
            self.states.append(polygon)
            
            


    def save_state(self):
        print("Saving State")
        self.states.append((deepcopy(self.current_polygon)) )

    def create_polygon(self, filename):
        label = LabelPolygon()
        label.creating = True
        
        self.current_polygon = label

        label.index = len(self.labels)
        self.labels.append(label)
        
        #create a file dictionary to keep track of which label exists to which file
        if filename in self.file_labels.keys():
            print("LABELS", self.file_labels.keys())
            # append a new label onto the filelist labels
            temp_labels = self.file_labels[filename]
            temp_labels.append(label.index)
            self.file_labels[filename] = temp_labels
        else:
            print("NEW LABEL")
            # new label encapsulated with list so it can append additional polygons later
            self.file_labels[filename] = [label.index] 
            print("NEW LABEL",label.index, self.file_labels.keys() , self.file_labels[filename])

        return label.index
        
    def export_label(self, polygon, participant, save_filename, load_file, overwrite=False):
        '''
        'Participant': Name of participant/user
        'Filename': Filename which polygon resides
        'Points':   Points uses to measure
        'Distance': Distance between first and last point (Euclidean) - important if used only 2 points to measure
        'TotalDistance': Distance between all points accumulated. - Important for measuring the distanece along the surface.                 
        '''

        filename = self.get_filename(polygon)
        points = polygon.points

        # Check if label already exists in csv, if it does skip writing this polygon.
        if os.path.isfile(save_filename) and not overwrite:
            open_csv = pd.read_csv(save_filename)

            # test_points = np.array([points])
            polygon_exists = ((open_csv['Participant'] == participant) & (open_csv['Filename'] == filename) & (open_csv['Points'] == str(points))).any()

            if polygon_exists:
                print("Polygon already exists by ",participant, " in ", filename, " : ", polygon)
                return


        # After checking if point does not exist, check if point_pairs are currently loaded.
        already_loaded = str(self.current_file) == str(load_file)
        if already_loaded:
            point_pairs = self.point_pairs
        else:
            loaded, image, depth, point_pairs,  points, colours = utils.load_projected(load_file)


        try:
            distance = utils.get_distance_2D(point_pairs, points[0], points[-1])
        except Exception as e:
            print(e)
            distance = 0

        total_distance = utils.get_total_distance(point_pairs, points)/10
        
        # filename = QFileDialog.getSaveFileName(None, 'Save File', "", "CSV (*.csv)")[0]
        # csv_path = filename.split('.')[0] + '.csv'
        # csv_path = participant + '.csv'



        
        data = [participant, filename, points, distance, total_distance]

        self.dataframe.loc[0] = data
        try:
            if os.path.isfile(save_filename):
                export_csv = self.dataframe.to_csv (save_filename, index = None, header=False, mode='a')
            else:
                export_csv = self.dataframe.to_csv (save_filename, index = None, header=True, mode='w')
            print("Save Complete")
        except Exception as e:
            print(e)
            print("Make sure you do not have", save_filename,  "open.")

    def export_all_labels(self, participant, current_folder, overwrite=False, use_previous_path=False):
        
        if self.export_path is None or use_previous_path == False:
            self.export_path = QFileDialog.getSaveFileName(None, 'Save File', "", "CSV (*.csv)")[0]

        for polygon in self.labels:

            filename = self.get_filename(polygon)
            load_file = utils.get_global_filename(current_folder, filename)
                
            self.export_label( polygon,  participant, self.export_path,load_file = load_file, overwrite=overwrite)

    
    def get_filename(self, polygon):
        filename = None
        for key in self.file_labels.keys():
            if self.labels.index(polygon) in self.file_labels[key]:
                filename = key
                break
        return filename
    
    def get_filename_by_index(self, polygon_index):
        filename = None
        for key in self.file_labels.keys():
            if polygon_index in self.file_labels[key]:
                filename = key
                break
        return filename

    def display_distance(self):
        points = self.current_polygon.points[-2:]
        dist = utils.get_distance(self.point_pairs, points[0], points[1])
        # print(dist)

    def get_polygon(self, index):
        return self.labels[index]
    
    def remove_point(self, photoviewer):
        photoviewer.redraw_image()
        self.save_state()
        
        point = self.current_polygon.check_near_point(photoviewer.mouse_pos, dist=2)
        self.current_polygon.remove_point(point)
        self.selected_point = None
        
        image = self.current_polygon.draw(photoviewer.image)
        photoviewer.update_image(image)

    def delete_polygon(self, index, next_index):
        # self.save_state()
        print(next_index)
        #removes index from filelist
        for filename in self.file_labels.keys():
            indices = self.file_labels[filename]

            if index in indices:
                new_indices = self.file_labels[filename]
                new_indices.remove(index)
                print("New indices", new_indices)
                self.file_labels[filename] = new_indices
                # print(self.file_labels[filename])

                # For every index greater than the removed index, reduce index by 1 if we remove an index
                # we use new indices or else we reach a segmentation fault
                for i in new_indices:
                    if i > index:
                        # Item no longer coorelates to index, so we find the item's index
                        item_index = self.file_labels[filename].index(i)
                        self.file_labels[filename][item_index] = self.file_labels[filename][item_index] -1
            

        
        if next_index == -1:
            self.current_polygon = None
        else:
            self.current_polygon = self.labels[next_index]
        self.labels.pop(index)

        # if we deleted all items from filename, delete the key
        if self.file_labels[filename] == []:
            self.file_labels.pop(filename)

        
        
        # self.current_polygon = index-1

    def load_ply(self, filename, progressbar=None):
        self.current_file = filename

        loaded, self.image, self.depth, self.point_pairs,  points, colours = utils.load_projected(filename, progressbar)
        if not loaded:
            print("Did not load projected...")
            plydata = PlyData.read(filename)
            print("Calculating Projection...")
            self.image, self.depth, self.point_pairs = utils.project_2D(utils.KINECT_AZURE_INTRINSICS, plydata, progress=progressbar)
            self.cloud, points, colours = utils.map_pairs_2D(self.image, self.point_pairs)
            

            utils.save_projected(filename, self.image, self.depth, self.point_pairs, points, colours)

        else:
            print("Projection loaded...")
            # progressbar.setFormat("Converting Pointcloud...")
            # progressbar.setValue(60)
            self.cloud = utils.numpy_to_o3d(np_cloud_points=points, np_cloud_colors=colours, swap_RGB=True)
            # progressbar.setValue(100)
            # progressbar.setFormat("Done...")
    
    def assign_polygon(self, index, points):
        for point in points:
            self.labels[index].assign_point(point)

    def measure_line_similarity(self, polygon_1, polygon_2, upper_limit=10):
        '''
        Measures similarity of the closest ends based on the relative distance formula

        relative difference = (x2 - x1)/x1
        Where x1 is the upper limit. 

        upper_limit is in mm

        Value of 0 is no difference, value of 1 is upper limit and value past that is more than upper limit in percent.
        '''
        # make sure both polygons have at least 2 points, otherwise we cannot measure
        if len(polygon_1.points) < 2 or len(polygon_2.points) < 2:
            print("Lengths not long enough", len(polygon_1.points), len(polygon_2.points))
            return None


        # Measure differences between all for points (start and end of both lines)
        # print(polygon_1.points[0], polygon_2.points[0])
        end_1 = utils.get_distance_2D(self.point_pairs, polygon_1.points[0], polygon_2.points[0])
        end_2 = utils.get_distance_2D(self.point_pairs, polygon_1.points[-1], polygon_2.points[-1])

        cross_1 = utils.get_distance_2D(self.point_pairs, polygon_1.points[0], polygon_2.points[-1])
        cross_2 = utils.get_distance_2D(self.point_pairs, polygon_1.points[-1], polygon_2.points[0])
        
        # Average distance between both points
        cross_distance = (cross_1 + cross_2)/2
        end_distance = (end_1 + end_2)/2

        # Match ends (order of operation affects this)
        # Score = Relative difference formula
        score = 0

        if cross_distance <= end_distance:
            score = cross_distance / upper_limit
        else:
            score = end_distance / upper_limit
        
        return abs(score)

    def measure_polygon_fit(self, line, polygon):
        p1 = line.points[0]
        p2 = line.points[-1]

        # Returns positive inside, negative outside and 0 on an edge
        p1_test = cv2.pointPolygonTest(np.array([polygon.points]), p1, False)
        p2_test = cv2.pointPolygonTest(np.array([polygon.points]), p2, False)

        if p1_test >= 0 or p2_test >=0:
            return True
        else:
            return False