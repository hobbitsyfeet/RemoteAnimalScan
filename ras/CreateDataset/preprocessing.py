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
        
def combine_ply_from_folder(file_list, labels):
    """
    File_list is a list of files and their paths
    Labels: a list of strings you want to extract based on labels

    Returns: 
            cloud_list = [o3d_cloud, o3d_cloud, ..., o3d_cloud] : List of o3d_Clouds
            labels_list = [[0,1,2, ... , N], ..., [0,1,2, ... , N]] : List of integer lists containing IDs


    This function iterates through all files in file list. It groups files by their folders and that folder creates a merged pointcloud.
    In the process, it takes the file name and grabs the labels from that, and creates a label_list containing a label for each point in the pointcloud
    
    """
    cloud_list = []
    label_list = []
    
    #Becasue we remove files from file_list as we fit them into groups, we want to iterate the folders until there is none left
    while file_list:
        for file in file_list:
            group = set()

            #Grabs the folder name
            folder = os.path.basename(os.path.dirname(file))
            
            #Searches entire list for files which are also in that folder
            for index, search_file in enumerate(file_list):
                search_folder = os.path.basename(os.path.dirname(search_file))

                if search_folder == folder:
                    #creates a group which contains all files in the one folder
                    group.add(search_file)

            #We now have every file in the group from the folder
            print("GROUP", group)

            #remove items in group from file_list
            for item in group:
                file_list.remove(item)
                        
            #labels of all pointclouds in group will be merged into merge_labels.
            merged_labels = []
            merged_cloud = o3d.geometry.PointCloud()

            for index, cloud in enumerate(group):
                #Load and join pointclouds
                pcd_cloud = o3d.io.read_point_cloud(cloud)
                merged_cloud = merged_cloud + pcd_cloud

                # Grab labels from the group like (0,"Head") from Head.ply
                group_labels = get_labels([cloud], labels) 
                print(group_labels)
                # as labels exist one-per-file, we extend the label to match points.
                merged_labels.extend([group_labels[0][0]]*get_num_points(pcd_cloud)) 

            # Create lists of merged clouds lists of labels according to the cloud
            cloud_list.append(merged_cloud)
            label_list.append(merged_labels)

    print(label_list)
    return cloud_list, label_list
    
            

        
def downsample(cloud, labels, method=VOXEL_SAMPLE, value=0.5, print_down_labels = False):
    sampled_labels = []

    if method == RANDOM_SAMPLE:
        random.seed(random.random())
        points, colors, normals = o3d_to_numpy(cloud)
        
        sampled_points = random.sample(points, value)
        sampled_colors = random.sample(colors, value)
        sampled_normals = random.sample(normals, value)

        sampled_labels = random.sample(labels, value)
        sparse_pcd = numpy_to_o3d(sampled_points, sampled_colors, sampled_normals)

        print("Before Downsample: ", cloud, end=" | ")
        print("After Downsample: Pointcloud with ", len(sparse_pcd[0].points), "points." )
    
    elif method == VOXEL_SAMPLE:
        # Downsample points
        min_bound = cloud.get_min_bound() - value * 0.5
        max_bound = cloud.get_max_bound() + value * 0.5
        
        #Old version
        # sparse_pcd, cubics_ids = cloud.voxel_down_sample_and_trace(value, min_bound, max_bound, False)
        print("Before Downsample: ", cloud, end=" | ")
        sparse_pcd = cloud.voxel_down_sample_and_trace(value, min_bound, max_bound, False)
        print("After Downsample: Pointcloud with ", len(sparse_pcd[0].points), "points." )

        cubics_ids = sparse_pcd[1]
        sparse_pcd = sparse_pcd[0]


        # Downsample labels
        # Solution from https://github.com/intel-isl/Open3D-PointNet2-Semantic3D/blob/master/downsample.py and modified.
        for cubic_id in cubics_ids:
            cubic_id = cubic_id[cubic_id != -1]

            cubic_labels = []
            for label in cubic_id:
                cubic_labels.append(labels[label])

            #Label is the maximum count of labels in voxel
            sampled_labels.append(np.bincount(cubic_labels).argmax())

            if print_down_labels:
                print("Cubic Labels", cubic_labels, end=" -> ")
                print(sampled_labels[-1])
            
    print("Sampled Labels", sampled_labels)

    return sparse_pcd, sampled_labels

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

def export_hdf5(filename, test_points, train_points, test_colors, train_colors, test_normals, train_normals, test_labels, train_labels, point_num_test, point_num_train, get_max_points):
    """
    test_list: list of pointclouds which are split into the test sample
    train_list: list of pointclouds which are split into the train sample

    test_labels: A list which contains per-point labels(int) for each pointcloud. Eg [cloud1[0,0,0 ... 0], cloud2[2,2 .. 2] ]. Each inner list contains per-point labels  
    """
    data_h5 = np.zeros((len(cloud_list), max_points, 3))
    colors_h5 = np.zeros((len(cloud_list), max_points, 3))
    normals_h5 = np.zeros((len(cloud_list), max_points, 3))
    labels_h5 = np.zeros((len(cloud_list), max_points))


    for index, cloud in enumerate(cloud_list):
        np_cloud_points, np_cloud_colors, np_cloud_normals = o3d_to_numpy(cloud)
        
        data_h5[:np_cloud_points.shape[0],:np_cloud_points.shape[1],:np_cloud_points.shape[2]] = np_cloud_points
        print(data_h5)
        # point_list.append(np_cloud_points)
        # color_list.append(np_cloud_colors)
        # normal_list.append(np_cloud_normals)



    test_points, train_points, test_colors, train_colors, test_normals, train_normals, test_labels, train_labels, point_num_test, point_num_train  = train_test_split(point_list, color_list, normal_list, cloud_labels, point_num_list, test_size=0.33, random_state=42)

    train_filename = filename +"_train.h5"
    print(train_filename)
    hdf_train = h5py.File(train_filename, "w")
    #dataset = f.create_dataset("data", data = point_data)
    hdf_train.create_dataset("data", data = train_points)
    hdf_train.create_dataset("data_num", data = point_num_train)
    # hdf_test.create_dataset("label", data = test_labels) #Here we are just saying the labels belong to only one object (stick man, raccoon...)
    hdf_train.create_dataset("label_seg", data = train_labels) #?
    hdf_train.create_dataset("color", data = train_colors)
    hdf_train.create_dataset("normal", data = train_normals)

    hdf_train.flush()
    hdf_train.close()

    test_filename = filename +"_test.h5"
    filename.write(test_filename + '\n')
    hdf_test = h5py.File(test_filename, "w")
    #dataset = f.create_dataset("data", data = point_data)
    hdf_test.create_dataset("data", data = test_points)
    hdf_test.create_dataset("data_num", data = point_num_test)
    # hdf_test.create_dataset("label", data = test_labels) #Here we are just saying the labels belong to only one object (stick man, raccoon...)
    hdf_test.create_dataset("label_seg", data = label_test) #?
    hdf_test.create_dataset("color", data = test_colors)
    hdf_test.create_dataset("normal", data = test_normals)


    hdf_test.flush()
    hdf_test.close()
    print("HDF5 DATASET COMPLETE")
    

def get_max_points(pointcloud_list):
    """
    """
    max_number = 0
    for cloud in pointcloud_list:
        #get the number of points in the pointcloud
        number = get_num_points(cloud) 

        if number > max_number:
            max_number = number
    return max_number

def get_num_points(cloud):
    return np.asarray(cloud.points).shape[0] 

def estimate_normals(cloud, radius=0.1, max_nn=30):

    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius, max_nn))
    # o3d.visualization.draw_geometries([downpcd])
    return cloud


def write_settings(self, setting, value):
    pass

