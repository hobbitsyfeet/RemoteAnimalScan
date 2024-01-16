# https://keras.io/examples/vision/pointnet_segmentation/
# https://github.com/soumik12345/point-cloud-segmentation
import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
print("Loading Tensorflow...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
print("Done loading Tensorflow")
import matplotlib.pyplot as plt






import h5py

# https://aclanthology.org/W17-2339.pdf
# In this example, the hierarchical labels are organized into a nested dictionary where the keys at the top level represent the classes, and the keys at the second level represent the subclasses. The values in the dictionary are integer labels that correspond to each subclass.

# The data points are defined as a NumPy array, where each row represents a single data point and each column represents a feature of the data.

# To train the model on the data points and hierarchical labels, we simply pass the data and labels to the fit() method of the Keras model. The model will then use the hierarchical structure of the labels to learn the relationships between the data points and their corresponding classes and subclasses.
# # Define the hierarchical labels as a nested Python dictionary
# hierarchical_labels = {
#     "class1": {
#         "subclass1": 0,
#         "subclass2": 1
#     },
#     "class2": {
#         "subclass1": 2,
#         "subclass2": 3
#     }
# }

# # Define the data points as a NumPy array
# data_points = np.array([
#     [1.0, 2.0, 3.0],
#     [4.0, 5.0, 6.0],
#     [7.0, 8.0, 9.0],
#     [10.0, 11.0, 12.0],
#     ...
# ])


# # Create a Keras model
# model = keras.Sequential()
# model.add(keras.layers.Dense(32, input_shape=(3,)))
# model.add(keras.layers.Dense(32, activation="relu"))
# model.add(keras.layers.Dense(16, activation="softmax"))

# # Compile the model
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# # Train the model on the data points and hierarchical labels
# model.fit(data_points, hierarchical_labels, epochs=10)

# # Create a Keras model
# model = keras.Sequential()
# model.add(keras.layers.Dense(32, input_shape=(3,)))
# model.add(keras.layers.Dense(32, activation="relu"))
# model.add(keras.layers.Dense(16, activation="softmax"))

# # Compile the model
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# # Train the model on the data points and hierarchical labels
# model.fit(data_points, hierarchical_labels, epochs=10)







# dataset_url = "https://git.io/JiY4i"

# dataset_path = keras.utils.get_file(
#     fname="shapenet.zip",
#     origin=dataset_url,
#     cache_subdir="datasets",
#     hash_algorithm="auto",
#     extract=True,
#     archive_format="auto",
#     cache_dir="datasets",
# )

# with open("/tmp/.keras/datasets/PartAnnotation/metadata.json") as json_file:
#     metadata = json.load(json_file)

# print(metadata)

# points_dir = "/tmp/.keras/datasets/PartAnnotation/{}/points".format(
#     metadata["Airplane"]["directory"]
# )
# labels_dir = "/tmp/.keras/datasets/PartAnnotation/{}/points_label".format(
#     metadata["Airplane"]["directory"]
# )
# LABELS = metadata["Airplane"]["lables"]
# COLORS = metadata["Airplane"]["colors"]

# VAL_SPLIT = 0.2
# NUM_SAMPLE_POINTS = 1024
# BATCH_SIZE = 32
# EPOCHS = 60
# INITIAL_LR = 1e-3

# point_clouds, test_point_clouds = [], []
# point_cloud_labels, all_labels = [], []

# points_files = glob(os.path.join(points_dir, "*.pts"))
# for point_file in tqdm(points_files):
#     point_cloud = np.loadtxt(point_file)
#     if point_cloud.shape[0] < NUM_SAMPLE_POINTS:
#         continue

#     # Get the file-id of the current point cloud for parsing its
#     # labels.
#     file_id = point_file.split("/")[-1].split(".")[0]
#     label_data, num_labels = {}, 0
#     for label in LABELS:
#         label_file = os.path.join(labels_dir, label, file_id + ".seg")
#         if os.path.exists(label_file):
#             label_data[label] = np.loadtxt(label_file).astype("float32")
#             num_labels = len(label_data[label])

#     # Point clouds having labels will be our training samples.
#     try:
#         label_map = ["none"] * num_labels
#         for label in LABELS:
#             for i, data in enumerate(label_data[label]):
#                 label_map[i] = label if data == 1 else label_map[i]
#         label_data = [
#             LABELS.index(label) if label != "none" else len(LABELS)
#             for label in label_map
#         ]
#         # Apply one-hot encoding to the dense label representation.
#         label_data = keras.utils.to_categorical(label_data, num_classes=len(LABELS) + 1)

#         point_clouds.append(point_cloud)
#         point_cloud_labels.append(label_data)
#         all_labels.append(label_map)
#     except KeyError:
#         test_point_clouds.append(point_cloud)

# print("Pointclouds", len(point_clouds))


# for i in range(5):
#     # i = random.randint(0, len(point_clouds) - 1)
#     print(f"point_clouds[{i}].shape:", point_clouds[0].shape)
#     print(f"point_cloud_labels[{i}].shape:", point_cloud_labels[0].shape)
#     for j in range(5):
#         print(
#             f"all_labels[{i}][{j}]:",
#             all_labels[i][j],
#             f"/tpoint_cloud_labels[{i}][{j}]:",
#             point_cloud_labels[i][j],
#             "/n",
#         )


COLORS = ['Red', 'Green', 'Blue', 'Orange', 'Purple', 'Yellow']
def visualize_data(point_cloud, labels):
    df = pd.DataFrame(
        data={
            "x": point_cloud[:, 0],
            "y": point_cloud[:, 1],
            "z": point_cloud[:, 2],
            "label": labels,
        }
    )
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection="3d")
    for index, label in enumerate(LABELS):
        c_df = df[df["label"] == label]
        try:
            ax.scatter(
                c_df["x"], c_df["y"], c_df["z"], label=label, alpha=0.5, c=COLORS[index]
            )
        except IndexError:
            pass
    ax.legend()
    plt.show()





# for index in tqdm(range(len(point_clouds))):
#     current_point_cloud = point_clouds[index]
#     current_label_cloud = point_cloud_labels[index]
#     current_labels = all_labels[index]
#     num_points = len(current_point_cloud)
#     # Randomly sampling respective indices.
#     sampled_indices = random.sample(list(range(num_points)), NUM_SAMPLE_POINTS)
#     # Sampling points corresponding to sampled indices.
#     sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
#     # Sampling corresponding one-hot encoded labels.
#     sampled_label_cloud = np.array([current_label_cloud[i] for i in sampled_indices])
#     # Sampling corresponding labels for visualization.
#     sampled_labels = np.array([current_labels[i] for i in sampled_indices])
#     # Normalizing sampled point cloud.
#     norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
#     norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
#     point_clouds[index] = norm_point_cloud
#     point_cloud_labels[index] = sampled_label_cloud
#     all_labels[index] = sampled_labels


# visualize_data(point_clouds[0], all_labels[0])
# visualize_data(point_clouds[300], all_labels[300])

NUM_SAMPLE_POINTS = 1024
LABELS = ['tail', 'arm', 'hand', 'head', 'body']

def load_data(point_cloud_batch, label_cloud_batch):
    point_cloud_batch.set_shape([NUM_SAMPLE_POINTS, 3])
    label_cloud_batch.set_shape([NUM_SAMPLE_POINTS, len(LABELS) + 1])
    return point_cloud_batch, label_cloud_batch


def augment(point_cloud_batch, label_cloud_batch):
    noise = tf.random.uniform(
        tf.shape(label_cloud_batch), -0.005, 0.005, dtype=tf.float64
    )
    point_cloud_batch += noise[:, :, :3]
    return point_cloud_batch, label_cloud_batch


def generate_dataset(point_clouds, label_clouds, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((point_clouds, label_clouds))
    dataset = dataset.shuffle(BATCH_SIZE * 100) if is_training else dataset
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = (
        dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training
        else dataset
    )
    return dataset


# split_index = int(len(point_clouds) * (1 - VAL_SPLIT))
# train_point_clouds = point_clouds[:split_index]
# train_label_cloud = point_cloud_labels[:split_index]
# total_training_examples = len(train_point_clouds)

# val_point_clouds = point_clouds[split_index:]
# val_label_cloud = point_cloud_labels[split_index:]

# print("Num train point clouds:", len(train_point_clouds))
# print("Num train point cloud labels:", len(train_label_cloud))
# print("Num val point clouds:", len(val_point_clouds))
# print("Num val point cloud labels:", len(val_label_cloud))

# train_dataset = generate_dataset(train_point_clouds, train_label_cloud)
# val_dataset = generate_dataset(val_point_clouds, val_label_cloud, is_training=False)

# print("Train Dataset:", train_dataset)
# print("Validation Dataset:", val_dataset)


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv1D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config

def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=64, name=f"{name}_1")
    x = conv_block(x, filters=128, name=f"{name}_2")
    x = conv_block(x, filters=1024, name=f"{name}_3")
    x = layers.GlobalMaxPooling1D()(x)
    x = mlp_block(x, filters=512, name=f"{name}_1_1")
    x = mlp_block(x, filters=256, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=(2, 1), name=f"{name}_mm")([inputs, transformed_features])


def get_shape_segmentation_model(num_points: int, num_classes: int) -> keras.Model:
    input_points = keras.Input(shape=(None, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)

def segmentation_and_keypoint_detection_network(num_points:int, num_classes:int, num_keypoints:int):
    # Input layer.
    input_points = keras.Input(shape=(None, 3))

    # PointNet Classification Network.
    transformed_inputs = transformation_block(
        input_points, num_features=3, name="input_transformation_block"
    )
    features_64 = conv_block(transformed_inputs, filters=64, name="features_64")
    features_128_1 = conv_block(features_64, filters=128, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=128, name="features_128_2")
    transformed_features = transformation_block(
        features_128_2, num_features=128, name="transformed_features"
    )
    features_512 = conv_block(transformed_features, filters=512, name="features_512")
    features_2048 = conv_block(features_512, filters=2048, name="pre_maxpool_block")
    global_features = layers.MaxPool1D(pool_size=num_points, name="global_features")(
        features_2048
    )
    global_features = tf.tile(global_features, [1, num_points, 1])

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            transformed_features,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    segmentation_output = layers.Conv1D(
        num_classes, kernel_size=1, activation="softmax", name="segmentation_head"
    )(segmentation_features)

    # Keypoint detection head.
    keypoint_input = global_features
    keypoint_features = conv_block(
        keypoint_input, filters=128, name="keypoint_features"
    )
    keypoint_output = layers.Conv1D(
        num_keypoints, kernel_size=1, activation="sigmoid", name="keypoint_head"
    )(keypoint_features)

    # Define the model.
    model = keras.Model(inputs=input_points, outputs=[segmentation_output, keypoint_output])
    return model

# x, y = next(iter(train_dataset))

# num_points = x.shape[1]
# num_classes = y.shape[-1]

# segmentation_model = get_shape_segmentation_model(num_points, num_classes)
# segmentation_model.summary()


# training_step_size = total_training_examples // BATCH_SIZE
# total_training_steps = training_step_size * EPOCHS
# print(f"Total training steps: {total_training_steps}.")

# lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
#     boundaries=[training_step_size * 15, training_step_size * 15],
#     values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25],
# )

# steps = tf.range(total_training_steps, dtype=tf.int32)
# lrs = [lr_schedule(step) for step in steps]

# plt.plot(lrs)
# plt.xlabel("Steps")
# plt.ylabel("Learning Rate")
# plt.show()


def run_experiment(epochs, train_dataset, val_dataset):

    segmentation_model = get_shape_segmentation_model(num_points, num_classes)
    segmentation_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )



    checkpoint_filepath = "/tmp/checkpoint/"
    if not os.path.exists(checkpoint_filepath):
        os.mkdir(checkpoint_filepath)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = segmentation_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    segmentation_model.load_weights(checkpoint_filepath)
    return segmentation_model, history





def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs") # ERROR Exception has occurred: KeyError       (note: full exception trace is shown but execution is paused at: <module>)
                            #'wing'
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()



def visualize_single_point_cloud(point_clouds, label_clouds, idx):
    label_map = LABELS + ["none"]
    point_cloud = point_clouds[idx]
    label_cloud = label_clouds[idx]
    visualize_data(point_cloud, [label_map[np.argmax(label)] for label in label_cloud])

def load_RAS_dataset(folder_path, level=0, type="hdf5"):
    """
    Loads a folder containing the structure output of RAS dataset. This dataset goes through the process of Label Dataset -> Create Dataset
    """
    
    
    if type == "hdf5":
        files = glob((folder_path+"*.h5"))
        training_path = [file for file in files if "train" in file] # Grabs the file with substring "train"
        testing_path = [file for file in files if "test" in file]
        training_dataset = h5py.File(training_path[0], 'r') 
        testing_dataset = h5py.File(testing_path[0], 'r')

        if level == 0:
            pass

        elif level == 1:
            pass
    
    return training_dataset, testing_dataset


def load_RAS_json():
    pass

def start():
    print("Starting Pointnet pipeline...")

    dataset_dir = "K:/Github/RemoteAnimalScan/Training Data/"
    training_dataset, testing_dataset = load_RAS_dataset(dataset_dir)
    training_data = training_dataset['points']
    # training_labels = training_dataset['label_seg']

    testing_data = list(testing_dataset['points'])
    # testing_labels = list(testing_dataset['label_seg'])
    # num_classes_training = np.unique(list(training_dataset['label_seg']))
    # num_classes_testing = np.unique(list(testing_dataset['label_seg']))
    num_classes = len(set().union(*[np.unique(list(training_dataset['label_seg'])), np.unique(list(testing_dataset['label_seg']))])) + 1

    training_labels = []

    for index, labels in enumerate(list(training_dataset['label_seg'])):
        label = keras.utils.to_categorical(labels.tolist(), num_classes=num_classes)
        training_labels.append(label)

    testing_labels = []
    # num_classes = len(np.unique(list(testing_dataset['label_seg']))) + 1
    for index, labels in enumerate(list(testing_dataset['label_seg'])):
        label = keras.utils.to_categorical(labels.tolist(), num_classes=num_classes)
        testing_labels.append(label)

    train_point_clouds = list(training_data)
    train_label_cloud = training_labels
    val_point_clouds = list(testing_data)
    val_label_cloud = testing_labels
    return train_point_clouds ,train_label_cloud, val_point_clouds, val_label_cloud

def generate_clustered_data(num_clusters, num_points, num_classes, num_keypoints):
    # Generate random cluster centers.
    centers = np.random.rand(num_clusters, 3) * 1 - 2
    cluster_size = int(num_points/num_clusters)

    # Generate random point cloud data.
    point_cloud_data = []
    segmentation_labels = []
    keypoint_labels = []
    for i in range(num_clusters):
        # Generate points around cluster center.
        # points = np.random.randn(num_points, 3) + centers[i]# Generate points around cluster center in a spiral shape.
        t = np.linspace(0, 10*np.pi, cluster_size)
        x = np.sin(t)
        y = np.cos(t)
        z = np.linspace(0, 10, cluster_size)
        points = np.column_stack((x, y, z)) + centers[i]
        point_cloud_data.append(points)

        # Generate random segmentation labels for each point in the cluster.
        segmentation_labels.append(np.full((cluster_size, 1), i % num_classes))

        # Generate random keypoint labels for each point in the cluster.
        keypoint_labels.append(np.random.rand(cluster_size, num_keypoints))

    point_cloud_data = np.array(point_cloud_data)
    segmentation_labels = np.array(segmentation_labels)
    keypoint_labels = np.array(keypoint_labels)

    print(centers)

    return point_cloud_data, segmentation_labels, keypoint_labels



# def plot_clustered_data(point_cloud_data, segmentation_labels):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     for i in range(segmentation_labels.shape[0]):
#         cluster_mask = segmentation_labels[i, :, 0] == 1
#         ax.scatter(point_cloud_data[i, cluster_mask, 0], point_cloud_data[i, cluster_mask, 1], point_cloud_data[i, cluster_mask, 2], label=f"Cluster {i}")

#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')

#     plt.legend()
#     plt.show()

def plot_clustered_data(point_cloud_data, segmentation_labels, keypoint_labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # for i in range(segmentation_labels.shape[1]):
    # cluster_mask = segmentation_labels[:, i, 0] == 1
    # keypoint_mask = np.argmax(keypoint_labels[cluster_mask], axis=1)
    

    # point_cloud_data[cluster_mask, 0]
    ax.scatter(point_cloud_data[:, 0], point_cloud_data[:, 1], point_cloud_data[:, 2])
    # ax.scatter(point_cloud_data[cluster_mask, 0], point_cloud_data[cluster_mask, 1], point_cloud_data[cluster_mask, 2], c=keypoint_mask/np.max(keypoint_mask), cmap='viridis')


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.legend()
    plt.show()

def load_RAS_json(file):
    

if __name__ == "__main__":

    # # Example input data.
    # # Example input data
    num_points = 1024
    num_classes = 4
    num_keypoints = num_classes

    # # Define the bodyparts to detect
    # bodyparts = ['left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    
    # # Generate some random point cloud data
    # point_cloud_data = np.random.rand(1, num_points, 3)
    # point_cloud_data

    # # Generate some random label data for segmentation and keypoints
    # segmentation_labels = np.random.randint(low=0, high=num_classes, size=(1, num_points, 1))
    # keypoint_labels = np.random.rand(1, num_points, num_keypoints)

    point_cloud_data, segmentation_labels, keypoint_labels = generate_clustered_data(num_classes, num_points, num_classes, num_keypoints)

    # # Plot data
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # for i in range(num_classes):
    #     cluster_mask = segmentation_labels[0,:,0] == i
    #     ax.scatter(point_cloud_data[0,cluster_mask,0], point_cloud_data[0,cluster_mask,1], point_cloud_data[0,cluster_mask,2])
    #     plt.show()
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # plt.show()

    plot_clustered_data(point_cloud_data, segmentation_labels, keypoint_labels)

    # Call the segmentation_and_keypoint_detection_network function with the example input data
    model = segmentation_and_keypoint_detection_network(num_points=num_points, num_classes=num_classes, num_keypoints=num_keypoints)
    # Train the model on the example input and label data
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy', 'binary_crossentropy'])
    model.fit(point_cloud_data, [segmentation_labels, keypoint_labels], epochs=500, batch_size=32)

    # Make predictions on a new point cloud
    new_point_cloud_data = np.random.rand(1, num_points, 3)
    segmentation_probs, keypoint_probs = model.predict(new_point_cloud_data)

    # Find the most likely position of each keypoint
# Create a dictionary to map the indices to the body part names
    part_names = {0: 'left_eye', 1: 'right_eye', 2: 'nose', 3: 'left_ear', 4: 'right_ear'}

    # Find the index of the highest predicted value for each keypoint
    max_indices = np.argmax(keypoint_probs, axis=1)

    # Create an empty list to store the predicted keypoints
    predicted_keypoints = []

    # Loop over the keypoints and find the corresponding body part name for each prediction
    for i, idx in enumerate(max_indices):
        body_part = part_names[i]
        predicted_keypoint = point_cloud_data[0, idx]
        predicted_keypoints.append((body_part, predicted_keypoint))
        
    # Print the predicted keypoints
    print(predicted_keypoints)
    exit()

    train_point_clouds ,train_label_cloud, val_point_clouds, val_label_cloud = start()
    visualize_single_point_cloud(train_point_clouds,train_label_cloud, 1)
    # dataset_url = "https://git.io/JiY4i"

    # dataset_path = keras.utils.get_file(
    #     fname="shapenet.zip",
    #     origin=dataset_url,
    #     cache_subdir="datasets",
    #     hash_algorithm="auto",
    #     extract=True,
    #     archive_format="auto",
    #     cache_dir="datasets",
    # )

    # with open("C:/Users/legom/Downloads/shapenet/PartAnnotation/metadata.json") as json_file:
    #     metadata = json.load(json_file)

    # print(metadata)

    # points_dir = "C:/Users/legom/Downloads/shapenet/PartAnnotation/{}/points".format(
    #     metadata["Airplane"]["directory"]
    # )
    # labels_dir = "C:/Users/legom/Downloads/shapenet/PartAnnotation/{}/points_label".format(
    #     metadata["Airplane"]["directory"]
    # )
    # LABELS = metadata["Airplane"]["lables"] # list ['wing' 'tail']
    # COLORS = metadata["Airplane"]["colors"] # ['red', 'green' 'blue']

    VAL_SPLIT = 0.2
    NUM_SAMPLE_POINTS = 512
    BATCH_SIZE = 5
    EPOCHS = 60
    INITIAL_LR = 1e-4

    # point_clouds, test_point_clouds = [], []
    # point_cloud_labels, all_labels = [], []

    # points_files = glob(os.path.join(points_dir, "*.pts"))
    # for point_file in tqdm(points_files):
    #     point_file = point_file.replace('\\', '/')
    #     point_cloud = np.loadtxt(point_file)
    #     if point_cloud.shape[0] < NUM_SAMPLE_POINTS:
    #         continue

    #     # Get the file-id of the current point cloud for parsing its
    #     # labels.
    #     file_id = point_file.split("/")[-1].split(".")[0]
    #     label_data, num_labels = {}, 0
    #     for label in LABELS:
    #         label_file = os.path.join(labels_dir, label, file_id + ".seg")
    #         if os.path.exists(label_file):
    #             label_data[label] = np.loadtxt(label_file).astype("float32") # This is a single list of labels {Wing:array[0,0,0,1,0,0,1]}
    #             num_labels = len(label_data[label]) # Number of labels in .seg file (seg_label)

    #     # Point clouds having labels will be our training samples.
    #     try:
    #         label_map = ["none"] * num_labels # List of None and fills out labels
    #         for label in LABELS:

    #             # label data 
    #             for i, data in enumerate(label_data[label]): # label = 'wing'
    #                 # label data one list of labels for each point
    #                 # ['wing: array[0,0,1,0,0]
    #                 #  'tail': array[1,0,0,0,0]

    #                 # ]

    #                 # Label map becomes a list of 'wing, tail' assignments per point instead of numbers

    #                 label_map[i] = label if data == 1 else label_map[i]
    #         label_data = [
    #             LABELS.index(label) if label != "none" else len(LABELS)
    #             for label in label_map
    #         ]

    #         # Apply one-hot encoding to the dense label representation.
    #         label_data = keras.utils.to_categorical(label_data, num_classes=len(LABELS) + 1)

    #         point_clouds.append(point_cloud)    # List of points (3d array = 1 pointcloud)
    #                                             # 0: Pointcloud
    #                                             # array([[ 0.09729, -0.082  , -0.00719],
    #                                             #     [-0.07732, -0.06545, -0.02463],
    #                                             #     [-0.0397 , -0.06545, -0.02103],
    #                                             #     ...,
    #                                             #     [ 0.02342, -0.06275, -0.21971],
    #                                             #     [ 0.04112, -0.06123, -0.14974],
    #                                             #     [-0.00064, -0.05845, -0.10217]])
    #                                             # 1:
    #                                             # array([[ 0.25047, -0.06912, -0.02763],
    #                                             #     [ 0.16421, -0.04005, -0.03361],
    #                                             #     [-0.22506,  0.00561, -0.0135 ],
    #                                             #     ...,
    #                                             #     [-0.01065, -0.05971, -0.12627],
    #                                             #     [ 0.30473, -0.04471,  0.0123 ],
    #                                             #     [ 0.21699, -0.06985, -0.02829]])



    #         point_cloud_labels.append(label_data)   # List of Label Ids
    #                                                 # 0:
    #                                                 # array([[0., 0., 0., 0., 1.],
    #                                                 #        [0., 1., 0., 0., 0.],
    #                                                 #        [0., 0., 0., 0., 1.],
    #                                                 #        ...,
    #                                                 #        [1., 0., 0., 0., 0.],
    #                                                 #        [1., 0., 0., 0., 0.],
    #                                                 #        [1., 0., 0., 0., 0.]], dtype=float32)
    #                                                 # 1:
    #                                                 # array([[0., 1., 0., 0., 0.],
    #                                                 #        [0., 1., 0., 0., 0.],
    #                                                 #        [0., 0., 1., 0., 0.],
    #                                                 #        ...,
    #                                                 #        [1., 0., 0., 0., 0.],
    #                                                 #        [0., 1., 0., 0., 0.],
    #                                                 #        [0., 1., 0., 0., 0.]], dtype=float32)
    #                                                 # 2:
    #                                                 # array([[1., 0., 0., 0., 0.],
    #         all_labels.append(label_map) # Label map is pointcloud[label_name] # Wing, None, None, Engine, etc
    #     except KeyError:
    #         test_point_clouds.append(point_cloud)

    # print("Pointclouds", len(point_clouds)) # List of numpy arrays shape [num_points, 3]


    # for i in range(5):
    #     # i = random.randint(0, len(point_clouds) - 1)
    #     print(f"point_clouds[{i}].shape:", point_clouds[0].shape)
    #     print(f"point_cloud_labels[{i}].shape:", point_cloud_labels[0].shape)
    #     for j in range(5):
    #         print(
    #             f"all_labels[{i}][{j}]:",
    #             all_labels[i][j],
    #             f"/tpoint_cloud_labels[{i}][{j}]:",
    #             point_cloud_labels[i][j],
    #             "/n",
    #         )


    # visualize_data(point_clouds[0], all_labels[0])
    # visualize_data(point_clouds[300], all_labels[300])

    # Sampling/Normalization
    # for index in tqdm(range(len(point_clouds))):
    #     current_point_cloud = point_clouds[index]
    #     current_label_cloud = point_cloud_labels[index]
    #     current_labels = all_labels[index]
    #     num_points = len(current_point_cloud)
    #     # Randomly sampling respective indices.
    #     sampled_indices = random.sample(list(range(num_points)), NUM_SAMPLE_POINTS)
    #     # Sampling points corresponding to sampled indices.
    #     sampled_point_cloud = np.array([current_point_cloud[i] for i in sampled_indices])
    #     # Sampling corresponding one-hot encoded labels.
    #     sampled_label_cloud = np.array([current_label_cloud[i] for i in sampled_indices])
    #     # Sampling corresponding labels for visualization.
    #     sampled_labels = np.array([current_labels[i] for i in sampled_indices])
    #     # Normalizing sampled point cloud.
    #     norm_point_cloud = sampled_point_cloud - np.mean(sampled_point_cloud, axis=0)
    #     norm_point_cloud /= np.max(np.linalg.norm(norm_point_cloud, axis=1))
    #     point_clouds[index] = norm_point_cloud
    #     point_cloud_labels[index] = sampled_label_cloud
    #     all_labels[index] = sampled_labels


    # split_index = int(len(point_clouds) * (1 - VAL_SPLIT))
    # train_point_clouds = point_clouds[:split_index] # Just takes the last 20% of pointclouds
    # train_label_cloud = point_cloud_labels[:split_index]
    # total_training_examples = len(train_point_clouds)

    # val_point_clouds = point_clouds[split_index:]
    # val_label_cloud = point_cloud_labels[split_index:]

    print("Num train point clouds:", len(train_point_clouds))
    print("Num train point cloud labels:", len(train_label_cloud))
    print("Num val point clouds:", len(val_point_clouds))
    print("Num val point cloud labels:", len(val_label_cloud))

    # train_point_clouds is a list of pointclouds shaped [1024, 3]
    # train_label_cloud is a list of labels shaped [1024, 5] (probably from hot vector)
    train_dataset = generate_dataset(train_point_clouds, train_label_cloud)
    val_dataset = generate_dataset(val_point_clouds, val_label_cloud, is_training=False)
    # print(list(training_dataset['label_seg']))
    # train_dataset = generate_dataset(list(training_dataset['data']), list(training_dataset['label_seg']), is_training=True)

    print("Train Dataset:", train_dataset)
    print("Validation Dataset:", val_dataset)


    x, y = next(iter(train_dataset))

    num_points = x.shape[1]
    num_classes = y.shape[-1]

    segmentation_model = get_shape_segmentation_model(num_points, num_classes)
    segmentation_model.summary()


    training_step_size = len(train_point_clouds) // BATCH_SIZE
    total_training_steps = training_step_size * EPOCHS
    print(f"Total training steps: {total_training_steps}.")

    lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[training_step_size * 15, training_step_size * 15],
        values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25],
    )

    steps = tf.range(total_training_steps, dtype=tf.int32)
    lrs = [lr_schedule(step) for step in steps]

    # plt.plot(lrs)
    # plt.xlabel("Steps")
    # plt.ylabel("Learning Rate")
    # plt.show()


    segmentation_model, history = run_experiment(epochs=EPOCHS, train_dataset=train_dataset, val_dataset=val_dataset)


    plot_result("loss")
    plot_result("accuracy")


    validation_batch = next(iter(val_dataset))
    val_predictions = segmentation_model.predict(validation_batch[0])
    print(f"Validation prediction shape: {val_predictions.shape}")

    idx = np.random.choice(len(validation_batch[0]))
    print(f"Index selected: {idx}")

    # Plotting with ground-truth.
    visualize_single_point_cloud(validation_batch[0], validation_batch[1], idx)

    # Plotting with predicted labels.
    visualize_single_point_cloud(validation_batch[0], val_predictions, idx)



