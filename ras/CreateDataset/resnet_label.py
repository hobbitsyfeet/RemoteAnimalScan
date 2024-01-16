print("Loading Tensorflow...")
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("GPU load successful.")
    except RuntimeError as e:
        print(e)
import pixellib
print("Successfully loaded Tensorflow")
from pixellib.custom_train import instance_custom_training
from pixellib.instance import custom_segmentation


# vis_img = instance_custom_training()
# vis_img.load_dataset("raccoons")
# vis_img.visualize_sample()
 

# train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes= 2, batch_size = 1)
# # train_maskrcnn.config.class_names=['BG', 'Wheat', 'Weed']
# train_maskrcnn.load_pretrained_model("F:/Github/Maskrcnn_Wheat/mask_rcnn_model.006-1.027496.h5")
# train_maskrcnn.load_dataset("wheat")

# train_maskrcnn.config.NUM_CLASSES = 4
# train_maskrcnn.config.IMAGE_META_SIZE=14

# train_maskrcnn.train_model(num_epochs = 300, augmentation=True, path_trained_models = "wheat/models")



class Label_Training():
    def __init__(self) -> None:

        self.prior_training = None

        self.train_maskrcnn = instance_custom_training()
        self.label_dataset_folder = None
        self.label_dataset_image = None
        self.last_trained_model = None
        # self.train_maskrcnn.load_pretrained_model("K:/Github/RemoteAnimalScan/mask_rcnn_coco.h5")

        
        # self.train_maskrcnn.modelConfig(network_backbone = , num_classes= 2, batch_size = 1)
        # self.train_maskrcnn.config.class_names=['BG', 'Wrist']
        # self.train_maskrcnn.load_dataset("K:/Github/RemoteAnimalScan/ras/CreateDataset/MRCNN/Dog")

        # self.train_maskrcnn.train_model(num_epochs = 50, augmentation=True, path_trained_models = "K:/Github/RemoteAnimalScan/ras/CreateDataset/MRCNN/Dog/Model")
        # print("Done training")

    # def load_prior_training(self, file)

    def get_unique_labels(dataset):
        pass

    def train(self, last_model=None, updated_dataset=None):

        
        self.train_maskrcnn.modelConfig(network_backbone = "resnet50", num_classes= 1, batch_size = 1)
        print("Loaded Config.")

        self.train_maskrcnn.load_pretrained_model("K:/Github/RemoteAnimalScan/mask_rcnn_coco.h5")
        print("Loaded Pretrained model")
        
        # self.train_maskrcnn.modelConfig(network_backbone = "resnet50", num_classes= 1, batch_size = 1)
        # self.train_maskrcnn.load_pretrained_model("K:/Github/RemoteAnimalScan/mask_rcnn_coco.h5")
        self.train_maskrcnn.load_dataset("K:/Github/RemoteAnimalScan/ras/CreateDataset/MRCNN/Dog")
        print("Loaded Dataset.")
        

        
        self.train_maskrcnn.config.class_names=['BG', 'Wrist']
        
        print("Starting to train model.")
        self.train_maskrcnn.train_model(num_epochs = 200, augmentation=True, path_trained_models = "K:/Github/RemoteAnimalScan/ras/CreateDataset/MRCNN/Dog/Model")
        
    def test(self, model ="K:/Github/RemoteAnimalScan/ras/CreateDataset/MRCNN//Dog/Model/mask_rcnn_model.004-0.239930.h5"):
        segment_image = custom_segmentation()
        segment_image.inferConfig(num_classes= 1, class_names= ["BG", "Wrist"])
        segment_image.load_model("K:/Github/RemoteAnimalScan/ras/CreateDataset/MRCNN/Dog/Model/mask_rcnn_model.187-0.014775.h5")
        segment_image.segmentImage("K:/Github/RemoteAnimalScan/ras/CreateDataset/MRCNN/Dog/test/01_04_2022-00_16_51_6000.jpg", show_bboxes=True, output_image_name="sample_out.jpg")
#     def labels_to_json():

# # train_maskrcnn.config.class_names=['BG', 'Wheat', 'Weed']

# # train_maskrcnn.load_dataset("wheat")
#     def 


if __name__ == "__main__":
    label_assist = Label_Training()
    # label_assist.train()
    label_assist.test()
    # print("Done training")