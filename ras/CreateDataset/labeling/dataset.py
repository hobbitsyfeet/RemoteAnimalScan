from labeling.polygon import LabelShape
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


from PIL import Image

import json
import uuid

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
        self.current_file = None
        self.image = None
        self.depth = None
        self.point_pairs = None
        self.point_labels = None
        self.inverse_pairs = None
        self.cloud = None


        self.label_map = {}

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

        # return not empty if at least one label has a point
        for label in self.labels:
            if label.points:
                return False
        
        # Returns empty if there exist no labels or no labels with points
        return True


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
        label = LabelShape()
        label.creating = True
        
        self.current_polygon = label

        label.index = len(self.labels)
        self.labels.append(label)
        
        #create a file dictionary to keep track of which label exists to which file
        if filename in self.file_labels.keys():
            # append a new label onto the filelist labels
            temp_labels = self.file_labels[filename]
            temp_labels.append(label.index)
            self.file_labels[filename] = temp_labels
        else:
            # new label encapsulated with list so it can append additional polygons later
            self.file_labels[filename] = [label.index] 

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
            loaded, image, depth, point_pairs,  other_points, colours = utils.load_projected(load_file)


        try:
            distance = utils.get_distance_2D(point_pairs, points[0], points[-1])
        except Exception as e:
            print(e)
            distance = 0

        total_distance = utils.get_total_distance(point_pairs, points)/10
        
        # filename = QFileDialog.getSaveFileName(None, 'Save File', "", "CSV (*.csv)")[0]
        # csv_path = filename.split('.')[0] + '.csv'
        # csv_path = participant + '.csv'



        
        data = [participant, filename, points, distance, total_distance, polygon.name]

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

    def get_polygon_by_id(self, polygon_id):
        '''
        Returns polygon based on it's id
        '''
        for polygon in self.labels:
            if polygon.id == polygon_id:
                return polygon

        return None

    def get_polygons_by_file(self, filename):
        # if filename in self.file_labels.keys():
        polygons = []
        print(self.file_labels)


        # Handles case where there are no labels in filename
        if filename not in self.file_labels.keys():
            test_basename = os.path.basename(filename)
            if test_basename in self.file_labels.keys():
                filename = test_basename
            else:
                return []

        polygon_indecies = self.file_labels[filename]
        for i in polygon_indecies:
            polygons.append(self.get_polygon(i))
        return polygons

    def get_filenames(self):
        return list(self.file_labels.keys())

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

    def load_cloud(self, filename, progressbar=None):
        self.current_file = filename

        if filename[-4:] == "json":
            json_dict = utils.load_json_pointcloud(filename)
            self.cloud = utils.json_dict_to_o3d(json_dict)
            self.image, self.depth, self.point_pairs = utils.project_2D_multithread(utils.KINECT_AZURE_INTRINSICS, json_dict, progress=progressbar, dtype="json")
            self.cloud, points, colours = utils.map_pairs_2D(self.image, self.point_pairs)
            utils.save_projected(filename, self.image, self.depth, self.point_pairs, points, colours)
            return True

        loaded, self.image, self.depth, self.point_pairs,  points, colours = utils.load_projected(filename, progressbar)
        if not loaded:
            print("Did not load projected...")
            plydata = PlyData.read(filename)
            print("Calculating Projection...")
            self.image, self.depth, self.point_pairs = utils.project_2D_multithread(utils.KINECT_AZURE_INTRINSICS, plydata, progress=progressbar)
            # self.image, self.depth, self.point_pairs = utils.project_2D(utils.KINECT_AZURE_INTRINSICS, plydata, progress=progressbar)
            print("Mapping pairs...")
            self.cloud, points, colours = utils.map_pairs_2D(self.image, self.point_pairs)
            
            utils.save_projected(filename, self.image, self.depth, self.point_pairs, points, colours)
            return True

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

    # def apply_labels_to_cloud(self):
    #     print(self.cloud)

    def map_point_labels(self, filename = None, unlabeled=LabelShape("Background")):
        '''
        Goes through each label and applies a label map similar to point_pairs but applies label_names to each 3D point for a given filename

        unlabeled assigns unlabeled points to None by default or a specified label
        '''
        if filename is None:
            filename = self.current_file

        # Grab all labels in file
        labels = self.get_polygons_by_file(filename)

        # For each label in the file, collect the data from that label
        for label in labels:
            print(label.name)

            dst2, coords, cloud = label.get_segment_crop(self.image, self.point_pairs, edit=False)
            points = np.asarray(cloud.points)
            colours = np.asarray(cloud.colors)

            # For some reason colours are scaled between 0-1 (segment_crop likely to blame)
            colours = utils.unscale_rgb_np(colours)

            # Now we iterate through all the points in that label
            for index, point in enumerate(points):

                # Grab the point and colours from that label, and make a tuple
                data = (tuple(points[index]), tuple(colours[index]))
                self.label_map[data] = label


                # # # if data already exists within a label_map (multiple labels)
                # if data in self.label_map.keys():

                #     # get the label from this data
                #     value = self.label_map[data]

                #     if value is None:
                #         # Will this ever be the case?
                #         self.label_map[data] = label

                # #     # Otherwise this point will have 2 labels
                # #     else:
                # #         value.append(label)
                # else:
                # self.label_map[data] = label


        # Apply default label to unlabeled points
        for point_3d in self.point_pairs.values():
            
            if point_3d not in self.label_map.keys():
                self.label_map[point_3d] = unlabeled
            
        print("Unique Labels: ", set(self.label_map))
        return self.label_map



    def export_ply_and_labels(self, filename, format="json", load_label_map=True):

        if load_label_map:
            # NOTE this is important because we need labelmap to work
            # map_point_labels looks through all labels and maps that label to the list of points they contain
            # All non-labeled points are assigned an "unlabeled" category
            self.map_point_labels()

        # export_ply
        point_list = []
        colour_list = []
        label_list = []
        label_types = []
        label_ids = []

        for data in self.label_map.keys():
            point, colour = data
            label = self.label_map[data]
            point_list.append(point)
            colour_list.append(colour)
            label_list.append(label.name)
            label_ids.append(label.id)
            label_types.append(label.type)
            


        # label_ids = np.array(utils.convert_list_to_int(label_ids))

        if format == "ply":
            points = np.array(point_list)
            colours = np.array(colour_list)
            labels = np.array(utils.convert_list_to_int(label_list)) # Need to do this, ply does not accept character labels

            utils.write_o3d_ply(filename, points, colours, labels=labels)
            # utils.write_o3d_ply(filename, points, colours)

        if format == "json":
            # NOTE: we convert  list of floats to strings because json does not like floats.
            # Which means loading it in we must convert strings to floats.
            # Similarly with label_ids which are currently uuids of entire label (per point)
            json_dict = {
                "points":list(map(str, point_list)),
                "colours":list(map(str, colour_list)),
                "labels":label_list,
                "label_id":list(map(str, label_ids)),
                "label_types":label_types
            }
            self.save_json(filename , json_dict)

    def save_json(self, filename, data_dictionary):
        # pass
        if os.path.exists(filename):
            print("Overwriting file...")
        
        #Json filename
        
        json_filename = filename.split(".")[0] + ".json"
        print(json_filename)
        # Writing to sample.json
        
        # Serializing json
        json_object = json.dumps(data_dictionary, indent=4)

        with open(json_filename, "w") as outfile:
            outfile.write(json_object)



    # def json_dict_to_dataset(self, json_dict):


    def save_coco_file(self, filename, folder, image):
        # save jpg img
        image = np.nan_to_num(image).astype(np.uint8)
        save_jpg = image[:,:,::-1] # BGR to RGB
        save_jpg = Image.fromarray(image)
        
        image_name = filename.split(".")[0] + "." + "jpg"
        save_jpg.save(image_name)
        
        label_dict  = {}
        flags = ""
        version = "1.0.0"
        height, width, dim =  image.shape

        # load all polygons from img
        polygons = self.get_polygons_by_file(filename)

        # Fix the case where the filename with paths are not saved in file_list
        if polygon == []:
            polygons = self.get_polygons_by_file(os.path.basename(filename))

        shape_list = []
        for polygon in polygons:
            shape = {}
            shape["label"] = polygon.name
            shape["points"] = polygon.points
            shape["group_id"] = None
            shape["shape_type"] = "polygon"
            shape_list.append(shape)
        
        file = {}
        file["version"] = version
        file["flags"] = flags
        file["shapes"] = shape_list
        file["imagePath"] = image_name
        file["ImageData"] = None
        file["imageHeight"] = height
        file["imageWidth"] = width

        print(file)
        with open((image_name[:-3] + "json"), 'w') as outfile:
            json.dump(file, outfile, ensure_ascii=False, indent=2)
            

if __name__ == "__main__":
    dataset = Dataset()
    dataset.load_cloud("K:/Github/RemoteAnimalScan/data/Testing/01_04_2022-00_07_49_9000.projected")


