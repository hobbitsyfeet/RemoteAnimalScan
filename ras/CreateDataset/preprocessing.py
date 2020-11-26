import open3d as o3d
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import h5py

def o3d_to_numpy(o3d_cloud):
    """
    Converts open3d pointcloud to numpy array
    """
    np_cloud_points = np.asarray(o3d_cloud.points)
    np_cloud_colors = np.asarray(o3d_cloud.colors)
    np_cloud_normals = np.asarray(o3d_cloud.normals)

    return np_cloud_points, np_cloud_colors, np_cloud_normals

def numpy_to_o3d(np_cloud_points, np_cloud_colors=None, np_cloud_normals=None):
    #create o3d pointcloud and assign it
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np_cloud_points)
    if np_cloud_colors is not None:
        o3d_cloud.colors = o3d.utility.Vector3dVector(np_cloud_colors)
    if np_cloud_normals is not None:
        o3d_cloud.normals = o3d.utility.Vector3dVector(np_cloud_normals)

    return o3d_cloud

#FEATURES
COLOR = 0
NORMAL = 1

#SAMPLING_METHODS
RANDOM_SAMPLE = 2
VOXEL_SAMPLE = 3

#NORMALIZATION_METHODS
MINMAX = 4

#NONE
NONE = None

    
def load_dataset(file_list=[]):
    #return pointcloud_list
    o3d_list = []

    for file in file_list:
        o3d_cloud = o3d.io.read_point_cloud(file)

        # #just points
        # if not COLOR in feature_list and not NORMAL in feature_list:
        #     points, _, _ = o3d_to_numpy(o3d_cloud)
        #     o3d_cloud = numpy_to_o3d(points)

        # #just normals
        # elif NORMAL in feature_list and COLOR not in feature_list:
        #     if o3d_cloud.has_normals():
        #         points, _, normals = o3d_to_numpy(o3d_cloud)
        #         o3d_cloud = numpy_to_o3d(points, np_cloud_normals=normals)


        # #just colours
        # elif COLOR in feature_list and NORMAL not in feature_list:
        #     points, colors, _ = o3d_to_numpy(o3d_cloud)
        #     o3d_cloud = numpy_to_o3d(points, np_cloud_normals=colors)

        o3d_list.append(o3d_list)

    return o3d_list

def get_labels(file_list, labels):
    """
    File list only contains full file path

    Returns a label for each file based on the filename. (If label exists in filename)
    ex: animal_head_20201201.ply would return tuple(1, head) if head was the first label (1 index)
    
    If a object is loaded and the label does not exist in the list (for example, animal_background_20201201.ply)
    the labels will return (0, "unclassified")

    NOTE: You do not need to explicitly define unclassified as a label.
    """
    label_list = []
    # print(labels)
    for file in file_list:
        filename = os.path.splitext(file)
        # print("Filename", filename)
        
        for index, item in enumerate(labels):
            #checks if label exists in filename
            if item.lower() in filename[0].lower():
                label_list.append((index+1, item))
                break
            # #If not, label it as unclassified
            # else:
            #     print("UNKNOWN ITEM", item, filename[0].lower())
            #     label_list.append((0, "unclassified"))
    
    return label_list
        
def combine_ply_from_folder(file_list,labels):
    """
    File_list is a list of files and their paths
    """
    cloud_list = []
    label_list = []
    while file_list:
        for file in file_list:
            group = set()

            #Grabs the folder name
            folder = os.path.basename(os.path.dirname(file))
            
            #Searches entire list for files which are also in that folder
            for index, search_file in enumerate(file_list):
                search_folder = os.path.basename(os.path.dirname(search_file))
                # print("SEARCH FOLDER", search_folder)
                if search_folder == folder:
                    
                    #creates a temperary group which contains all files in the one folder
                    # group.append(search_file)
                    group.add(search_file)
                    #removes the file from file_list since it is already processed
                    # print(search_folder)
                    # file_list.pop(index)

            print("GROUP", group)
            #We now have every file in the group from the folder
            # print(group)

            #Grab labels from the group like (0,"Head") from Head.ply
            # labels = get_labels(group, label_list)
            print("LABELS", labels)
            merged_labels =[]

            for item in group:
                file_list.remove(item)
        #create a o3d_cloud based on all the points
        # merged_cloud = o3d.geometry.PointCloud()
        # for index, cloud in enumerate(group):
        #     cloud = o3d.io.read_point_cloud(cloud)
        #     merged_cloud += cloud

        #     #as labels exist one-per-file, we extend the label to match points.
        #     merged_labels.extend([labels]*get_num_points(cloud))

        # cloud_list.append(merged_cloud)
        # label_list.append(merged_labels)

    return cloud_list, label_list
    
            

        
def downsample(cloud, labels, method=VOXEL_SAMPLE, value=0.5):
    sampled_labels = []

    if method == RANDOM_SAMPLE:
        pass
    
    elif method == VOXEL_SAMPLE:
        # Downsample points
        min_bound = cloud.get_min_bound() - value * 0.5
        max_bound = cloud.get_max_bound() + value * 0.5

        sparse_pcd, cubics_ids = cloud.voxel_down_sample_and_trace(
            cloud, value, min_bound, max_bound, False
        )

        # Downsample labels
        # Solution from https://github.com/intel-isl/Open3D-PointNet2-Semantic3D/blob/master/downsample.py
        for cubic_id in cubics_ids:
            cubic_id = cubic_id[cubic_id != -1]
            cubic_labels = labels[cubic_id]
            #Label is the maximum count of labels in voxel
            sampled_labels.append(np.bincount(cubic_labels).argmax())

        # sampled_labels = np.array(sampled_labels)

        # cloud = cloud.voxel_down_sample(voxel_size=value)

    return cloud, sampled_labels

def get_bounding_box(cloud):
    pass
    
def normalize(cloud, method=MINMAX):
    
    # for index, cloud in enumerate(pointcloud_list):
    if method == MINMAX:
        
        #Extract the points into numpy
        points, colors, normals = o3d_to_numpy(cloud)

        #normalize the points only
        scaler = MinMaxScaler()
        scaler.fit(points)
        points = scaler.transform(points)

        #put the normalize pointcloud back into open3d, and into the list
        cloud = numpy_to_o3d(points, colors, normals)

    return cloud

def test_train_split(pointcloud_list, test_split=33, seed=42):
    """
    Test Split is how much (in percent) of the dataset should be turned into a testing dataset.
    Training dataset will be the remaining of the split value
    Seed is the test/train split seed. If the same seed is used, the test-train splits will be the same (if the same split value)
    42 is default because it's the answer to life and everything
    """
    pass

def export_hdf5(filename, test_list, train_list, test_labels, train_labels, point_num_test, point_num_train, max_points):
    """
    test_list: list of pointclouds which are split into the test sample
    train_list: list of pointclouds which are split into the train sample

    test_labels: A list which contains per-point labels(int) for each pointcloud. Eg [cloud1[0,0,0 ... 0], cloud2[2,2 .. 2] ]. Each inner list contains per-point labels  
    """

    train_filename = filename +"train.h5"
    hdf_train = h5py.File(train_filename, "w")
    #dataset = f.create_dataset("data", data = point_data)
    hdf_train.create_dataset("data", data = train_list)
    hdf_train.create_dataset("data_num", data = data_num_train)
    hdf_train.create_dataset("label", data = model_id_train) #Here we are just saying the labels belong to only one object (stick man, raccoon...)
    hdf_train.create_dataset("label_seg", data = label_train) #The labels for each point
    hdf_train.create_dataset("normal", data = normal_train) #surface normals

    hdf_train.flush()
    hdf_train.close()

    test_filename = filename +"_test.h5"
    filename.write(test_filename + '\n')
    hdf_test = h5py.File(test_filename, "w")
    #dataset = f.create_dataset("data", data = point_data)
    hdf_test.create_dataset("data", data = data_test)
    hdf_test.create_dataset("data_num", data = data_num_test)
    hdf_test.create_dataset("label", data = model_id_test) #Here we are just saying the labels belong to only one object (stick man, raccoon...)
    hdf_test.create_dataset("label_seg", data = label_test)
    hdf_test.create_dataset("normal", data = normal_test)

    hdf_test.flush()
    hdf_test.close()
    print("HDF5 DATASET COMPLETE")
    

def get_max_points(pointcloud_list):
    """
    """
    max_number = 0
    for cloud in pointcloud_list:
        #get the number of points in the pointcloud
        number = np.asarray(cloud.points).shape[0] 

        if number > max_number:
            max_number = number
    return max_number

def get_num_points(cloud):
    return np.asarray(cloud.points).shape[0] 

def estimate_normals(cloud, radius=0.1, max_nn=30):

    # o3d.geometry.estimate_normals(
    #     cloud,
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
    #                                                       max_nn=30))
    return cloud


def write_settings(self, setting, value):
    pass

