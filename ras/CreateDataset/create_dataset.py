import os
import sys

import open3d as o3d
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QIntValidator
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                             QComboBox, QDockWidget, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSpinBox, QStackedLayout, QStyle,
                             QTextEdit, QVBoxLayout, QWidget, QMenuBar)

import preprocessing as prep
from sklearn.model_selection import train_test_split

class App(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.unfiltered_files = None
        self.unfiltered_accepted = None

    def initUI(self):
        vbox = QVBoxLayout()
        header = QHBoxLayout()
        body = QHBoxLayout()
        body_right = QVBoxLayout()

        
        vbox.addLayout(header)
        vbox.addLayout(body)


        body_center = QVBoxLayout()
        
        self.accept_file_list = {}
        self.accept_file_widget = QListWidget()
        self.accept_file_widget.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.accept_file_widget.setUniformItemSizes(True)
        self.accept_file_widget.doubleClicked.connect(lambda: self.remove_files(self.accept_file_widget))
        self.setLayout(vbox)
        self.setGeometry(500, 500, 500, 400)
        self.setWindowTitle('Create Dataset')
        self.setWindowIcon(QIcon('web.png'))

        self.show()

        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.file_list_widget.doubleClicked.connect(lambda: self.add_files(self.file_list_widget, self.accept_file_widget))
        self.file_list = {}

        filter_layout = QHBoxLayout()
        self.filter = QLineEdit()
        self.filter.textChanged.connect(lambda: self.filter_list(self.file_list_widget, self.filter.text()))
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(lambda: self.filter_list(self.file_list_widget, self.filter.text() ))
        filter_layout.addWidget(self.filter_button)
        filter_layout.addWidget(self.filter)

        filter_accept_layout = QHBoxLayout()
        self.filter_accept = QLineEdit()
        self.filter_accept.textChanged.connect(lambda: self.filter_list(self.accept_file_widget, self.filter_accept.text()))
        self.filter_accept_button = QPushButton("Filter")
        self.filter_accept_button.clicked.connect(lambda: self.filter_list(self.accept_file_widget, self.filter_accept.text() ))
        filter_accept_layout.addWidget(self.filter_accept_button)
        filter_accept_layout.addWidget(self.filter_accept)
        
        self.folder_explorer = QLineEdit()
        self.folder_button = QPushButton("Browse")
        self.folder_button.clicked.connect(lambda: self.get_folder())


        self.add_button = QPushButton(icon=self.style().standardIcon(QStyle.SP_ArrowRight))
        self.add_button.clicked.connect(lambda: self.add_files(self.file_list_widget, self.accept_file_widget))
        # self.add_button.setIcon()
        self.remove_button = QPushButton(icon=self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.remove_button.clicked.connect(lambda: self.remove_files(self.accept_file_widget))

        self.visualize_button = QPushButton("Visualize")
        self.visualize_button.clicked.connect(lambda: self.visualize(self.accept_file_widget.selectedItems()))
        
        self.export_button = QPushButton("Export Dataset")
        self.export_button.clicked.connect(lambda: self.open_exporter(self.get_accepted_files()))

        header.addWidget(self.folder_explorer)
        header.addWidget(self.folder_button)
        

        
        body_left = QVBoxLayout()
        body.addLayout(body_left)

        body_left.addWidget(QLabel("File Explorer"))
        body_left.addWidget(self.file_list_widget)
        body_left.addLayout(filter_layout)

        body_center.addWidget(self.add_button)
        body_center.addWidget(self.remove_button)
        body.addLayout(body_center)
        body.addLayout(body_right)
        body_right.addWidget(QLabel("Dataset Files"))
        body_right.addWidget(self.accept_file_widget)
        body_right.addLayout(filter_accept_layout)
        
        body_right.addWidget(self.visualize_button)
        vbox.addWidget(self.export_button)
        
        # vbox.addWidget(self.file_list)
        
        # self.layout.addWidget(self.file_list)

    # def filterList(self, text, file_list):
        
    #     self.file_list.clear() # clear the list widget
    #     for file in file_list:
    #         if text in file: # only add the line if it passes the filter
    #             self.file_list.append(file)


    def init_menu(self):
        bar = QMenuBar(self)
        file = bar.addMenu("File")

    def get_folder(self):
        try:
            folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            # for file in folder:
            self.folder_explorer.setText(folder)
            self.file_list_widget.clear()
            self.file_list_widget.addItems(os.listdir(folder))
        except:
            print("No path")


    def add_files(self, from_list, to_list):
        items = from_list.selectedItems()

        widgets = []
        for item in items:
            if type(item) is not str:
                if os.path.isdir((self.folder_explorer.text() + "/" + item.text())):
                    print("TRUE")
                    items.extend(os.listdir((self.folder_explorer.text() + "/" + item.text())))
                    continue

            
            new_item = QListWidgetItem()
            if type(item) is not str:
                new_item.setData(0, item.text())
                new_item.setData(1, (self.folder_explorer.text() + "/" + item.text()))
            else:
                new_item.setData(0, item)
                new_item.setData(1, (self.folder_explorer.text() + "/" + item))
            new_item.setSizeHint(QSize(0, 17))
            widgets.append(new_item)
            item = None

        
        for widget in widgets:
            to_list.addItem(widget)

    
    def remove_files(self, file_list):
        items = file_list.selectedIndexes()
        for item in reversed(items):
            print(item.row())
            file_list.takeItem(item.row())

        # for index in range(len(files)):
        #     to_list.addItem(paths[index])
        # to_list.addItems()
        # to_list.addItems(1,paths)
            # to_list.addItems(filtered_list)
        # else:
        #     to_list.addItems(widget_items)


            
            # print(item.data(1))
            # print(to_list.findItems(paths))
        # print(filtered_list)
        
    def visualize(self, pcd_path):
        print(pcd_path)
        pointclouds = []
        for path in pcd_path:
            pcd = o3d.io.read_point_cloud(path.data(1),format="ply",print_progress=True)
            pointclouds.append(pcd)
        o3d.visualization.draw_geometries(pointclouds)

    def get_accepted_files(self):
        file_list = []
        for index in range(self.accept_file_widget.count()):
            item = self.accept_file_widget.item(index)
            file_list.append(item.data(1))
        return file_list

    def open_exporter(self, file_list):
        self.exporter = Exporter(file_list)
        self.exporter.show()

    def filter_list(self, file_list, filter_string):
        
        for index in range(file_list.count()):
            item = file_list.item(index)

            if filter_string == "":
                item.setHidden(False)
                continue
            print(item.data(0))
            print(filter_string)
            if filter_string in item.data(0):
                item.setHidden(False)
            else:
                item.setHidden(True)
            


class Exporter(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it 
    will appear as a free-floating window as we want.
    """
    def __init__(self, file_list):
        super().__init__()
        self.file_list = file_list
        self.setWindowTitle('Exporter')

        layout = QVBoxLayout()
        self.setLayout(layout)
        body = QHBoxLayout()
        body_right = QVBoxLayout()
        
        
        feature_group = QGroupBox("Features")
        feature_layout = QHBoxLayout()
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)

        layout.addLayout(body)

        self.normals = QCheckBox("Normals")
        self.colours = QCheckBox("Colours")

        feature_layout.addWidget(self.normals)
        feature_layout.addWidget(self.colours)

        points_group = QGroupBox("Points")
        points_layout = QVBoxLayout()
        points_group.setLayout(points_layout)
        body.addWidget(points_group)

        points_line = QStackedLayout()
        

        self.points = QSpinBox()
        self.points.setMaximum(999999999)
        self.points.setMinimum(1)
        self.points.setMaximumHeight(25)

        self.voxel_size = QDoubleSpinBox()
        self.voxel_size.setMaximum(999999999)
        self.voxel_size.setMinimum(0)
        self.voxel_size.setMaximumHeight(25)

        
        self.downsample = QComboBox()
        self.downsample.activated.connect(lambda: points_line.setCurrentIndex(self.downsample.currentIndex()))
        self.downsample.addItems(["Random Sample (# Points)", "Voxel Grid (m^3)"])
        points_layout.addWidget(self.downsample)
        points_layout.addLayout(points_line)

        scale_layout = QHBoxLayout()
        self.scale = QComboBox()
        self.scale.addItems(["Min-Max","Unit", "None"])
        scale_layout.addWidget(QLabel("Scale (normalize)"))
        scale_layout.addWidget(self.scale)
        points_layout.addLayout(scale_layout)

        self.standardize = QComboBox()
        self.standardize.addItems([])

        # points_line.(QLabel("Downsample: "))
        points_line.addWidget(self.points)
        points_line.addWidget(self.voxel_size)
        
        body.addLayout(body_right)
        labels_group = QGroupBox("Labels")
        
        body_right.addWidget(labels_group)
        
        
        labels_layout = QHBoxLayout()
        labels_vertical = QVBoxLayout()
        labels_group.setLayout(labels_vertical)
        labels_vertical.addLayout(labels_layout)

        self.labels_list = QListWidget()
        self.labels_list.doubleClicked.connect(lambda: self.remove_label(self.labels_list))

        
        self.labels_line = QLineEdit()
        self.labels_line.returnPressed.connect(lambda: self.add_label(self.labels_line.text(), self.labels_list))
        self.labels_line.returnPressed.connect(lambda: self.labels_line.setText(""))

        self.add_label_button = QPushButton("Add")
        self.add_label_button.clicked.connect(lambda: self.add_label(self.labels_line.text(), self.labels_list))
        self.add_label_button.clicked.connect(lambda: self.labels_line.setText(""))

        labels_layout.addWidget(self.labels_line)
        labels_layout.addWidget(self.add_label_button)

        labels_vertical.addWidget(self.labels_list)

        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(lambda: self.export(file_list))
        body.addWidget(self.export_button)

    def add_label(self, label, item_list):
        item_list.addItem(label)
    
    def remove_label(self, item_list):
        items = item_list.selectedIndexes()
        for item in reversed(items):
            print(item.row())
            item_list.takeItem(item.row())

    def get_items(self, widget):
        file_list = []
        for index in range(widget.count()):
            item = widget.item(index)
            file_list.append(item.data(0))
        return file_list

    def save_file(self):
        try:
            file_name = str(QFileDialog.getSaveFileName(self, "Select Directory"))
            # for file in folder:
        except:
            print("No path")
            pass
        return file_name

    def export(self, file_list):
        
        save_file = self.save_file()
        # Check which features should be loaded
        features = []
        if self.colours.isChecked():
            features.append(prep.COLOR)
        if self.normals.isChecked():
            features.append(prep.NORMAL)
        
        norm_method = self.scale.currentText()
        if norm_method == "MinMax":
            norm_method = prep.MINMAX
        elif norm_method == "Unit":
            norm_method = prep.NONE
        elif norm_method == "None":
            norm_method = prep.NONE

        down_method = self.downsample.currentIndex()
        down_value = 0
        if down_method == 0:
            down_method = prep.RANDOM_SAMPLE
            down_value = self.points.value()
        elif down_method == 1:
            down_method = prep.VOXEL_SAMPLE
            down_value = self.voxel_size.value()

        #Load all the Pointclouds from file list
        # cloud_list = prep.load_dataset(file_list)
        
        label_list = prep.get_labels(file_list, self.get_items(self.labels_list))
        # print(self.get_items(self.labels_list))
        print(label_list)
        cloud_list, cloud_labels = prep.combine_ply_from_folder(file_list, self.get_items(self.labels_list))
        # print(cloud_labels)
        point_num_list = []
        #Prepare data in dataset
        for index, cloud in enumerate(cloud_list):

            #Normalize
            cloud = prep.normalize(cloud, norm_method)

            #Downsample
            cloud, labels = prep.downsample(cloud, label_list[index], norm_method, down_value)

            #Grab Features (Normals)
            cloud = prep.estimate_normals(cloud)

            #?Normalize Colour?
            
            #a list describing how many points exist in each pointcloud
            point_num_list.append(prep.get_num_points(cloud))

            #Assign cloud to cloud_list, updating changes
            cloud_list[index] = cloud
            label_list[index] = labels[0]
        
        # print(cloud_list)
        # print(label_list)


        # #Get the maximum number of points in the dataset
        # max_points = prep.get_max_points(cloud_list)

        # #TestTrain Split
        # test_clouds, train_clouds, test_labels, train_labels, point_num_test, point_num_train = train_test_split(cloud_list, label_list, point_num_list, test_size=0.33, random_state=42)
        
        # #Export
        # prep.export_hdf5(save_file, test_clouds, train_clouds, test_labels, train_labels, point_num_list, max_points)




        


def main():
    app = QApplication(sys.argv)
    example = App()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
