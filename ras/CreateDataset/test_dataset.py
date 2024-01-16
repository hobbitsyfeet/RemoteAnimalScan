from labeling import dataset
import label_dataset
from PyQt5.QtWidgets import QApplication
import sys

import numpy as np
import open3d as o3d
from labeling import utils

from preprocessing import json_preprocess

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app = label_dataset.App()
    # d = dataset.Dataset()
    # app.dataset.load_cloud("K:/Github/RemoteAnimalScan/data/Training/01_04_2022-00_07_49_1000.projected")
    # 
    # app.load_labels(False, app.dataset, "K:/Github/RemoteAnimalScan/TestBuddy.csv")

    app.dataset.current_file = "K:/Github/RemoteAnimalScan/data/DuckSamples/_S_6000_Best.json"
    # app.dataset.load_cloud()

    # app.dataset.current_file="01_04_2022-00_07_49_1000.projected"
    # print(app.dataset.file_labels)
    # app.dataset.labels[0].name = "nose"
    # app.dataset.labels[1].name = "eye"
    # app.dataset.labels[2].name = "ear"
    # app.dataset.labels[3].name = "head"
    # app.dataset.labels[4].name = "something"
    # app.dataset.labels[5].name = "forearm"
    # app.dataset.labels[6].name = "forearm"
    # app.dataset.labels[7].name = "foot"
    # app.dataset.labels[8].name = "foot"
    # app.dataset.labels[9].name = "dog"


    # cloud_labels = app.dataset.map_point_labels()
    # # print(cloud_labels)

    # app.dataset.export_ply_and_labels(app.dataset.current_file)
    # # print(cloud_labels)
    print("loading json")
    json_dict = utils.load_json_pointcloud((app.dataset.current_file.split(".")[0] + ".json"))
    print("done")
    # print(json_dict)
    # pointclouds = []
    # pointclouds.append(utils.json_dict_to_o3d(json_dict))
    # o3d.visualization.draw_geometries(pointclouds)



    #  # get total point size
    # point_count = len(json_dict['points'])

    # # separate keypoints and segmentation for labelling
    # types = json_dict['label_types']

    # pointcloud_indices = utils.generate_indices(json_dict['points'])

    # keypoint_indices = utils.data_indices(types, 1)

    # # remove keypoints from set of global indices
    # non_keypoint_indices = list(set(pointcloud_indices).difference(set(keypoint_indices.tolist())))

    # # We do not want to sample keypoint indices so to keep sample sample size the same as value passed in we only sample non-keypoints and add keypoints back in
    # segment_sample_size = 1024 - len(keypoint_indices)
    
    # non_keypoint_sampled_indices = utils.random_sample_indices(segment_sample_size, seed=42)

    # non_keypoint_sampled = utils.select_indices(json_dict['points'], non_keypoint_sampled_indices)
    # keypoints = utils.select_indices(json_dict['points'], keypoint_indices)

    # # join keypoints into sample
    # total_sample = np.vstack((non_keypoint_sampled, keypoints))

    sampled_dict = json_preprocess(file_list=[json_dict])

    print(sampled_dict)

