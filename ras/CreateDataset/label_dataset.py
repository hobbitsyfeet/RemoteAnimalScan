# NOTE: Things to fix: 

# Get ./data folder loading automatically, or at least ask for folder to work. (DONE)
# Get rid of blue bar #DONE
# Locate where the data saves #DONE
# Zoom (Bind it to plus/minus as alternative) # DONE
# Command z for deleting last interacted point # DONE
# Redo button

# import open3d
import cv2
import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QAbstractItemView, QApplication, QAction, QCheckBox,
                             QComboBox, QDockWidget, QDoubleSpinBox, QMessageBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSpinBox, QStackedLayout, QStyle,
                             QTextEdit, QVBoxLayout, QWidget, QMenuBar, QGraphicsView, QToolButton, QGraphicsScene, QGraphicsPixmapItem, QFrame, QInputDialog, QProgressBar)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import QEvent
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import QThread, QObject, pyqtSignal

from copy import deepcopy

import os
# from ras.CreateDataset.labeling.dataset import Dataset

# from sklearn import cross_decomposition
# from ras.CreateDataset.utils import o3d_add_object
from labeling import utils

import random

from viewer import Action_Poll
# import label_scores

import pandas as pd

from labeling.dataset import Dataset
from labeling.scores import LabelScore


import webbrowser # To open help menu
# import pandas as pd
'''
######################################################################
CLASS APP


######################################################################
'''

# Labeling modes
LABEL_POLYGON = 0
LABEL_LINE = 1
LABEL_RECTANGLE = 2
LABEL_ELIPSE = 3

LOAD_ORDER_SEED = random.seed("DUCKS")

#class App(QWidget):
class App(QMainWindow):
    stop_signal = pyqtSignal()  # make a stop signal to communicate with the worker in another thread
    def __init__(self):
        super().__init__()

        # self.
        self.image = None
        # self.graphicsView.setMouseTracking(True)
        self.setMouseTracking(True)
        self.title = 'Label Image'
        self.setWindowIcon(QIcon('Logo_Final (3).png'))
        self.left = 50
        self.top = 50
        self.width = 1280
        self.height = 720
        self.zoom_height = 480
        self.autosave = True

        self.file_list = []
        self.selected_file = ""

        self.edit_mode = False
        self.label_mode = LABEL_LINE
        self.mouse_pos = (0,0)

        self.hover_point = None
        self.selected_point = None

        self.training_mode = False
        # self.labeled_scores = label_scores() # Scoring object to record progress made for practice labelling
        # self.current_polygon = None

        self.dataset = Dataset()
        self.training_datast = Dataset()
        self.testing_dataset = Dataset()

        self.coco_labels = None

        self.scores = LabelScore()
        self.test_set = self.training_datast

        self.load_order = None
        self.load_order_index = 0

        self.label_layouts = []

        self.o3d_vis = None
        # self.o3d = None
        self.threads = []
        self.worker = None

        self.init_buttons()
        self.initUI()
        
        self.user , pressed = QInputDialog.getText(self, "User Name", "Name: ",
                                           QLineEdit.Normal, "")
        while self.user == "":
            self.user , pressed = QInputDialog.getText(self, "User Name", "Name: ",
                                           QLineEdit.Normal, "")
        
        try:
            # path = os.path.abspath("./data/")
            # print(os.getcwd())
            path = os.path.expanduser(os.getcwd()) + "/data/Training/"
            print(path)
            self.get_folder(path)
            
        except Exception as e:
            print(e)
            print("Could not load data folder, find it by yourself.")
        
        self.load_seed = self.user

        # self.load_random(True, seed=self.user)

        self.load_labels(show_ui=False, dataset=self.training_datast, path=(os.path.expanduser(os.getcwd()) + "/data/Finished_GroundTruth.csv"))
        
        # self.file_list_widget.setSelectionMode(
        #     QAbstractItemView.ExtendedSelection
        # )
    def load_image():
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.o3d_vis is not None:
                self.stop_thread()

    @QtCore.pyqtSlot()
    def show_image(self, image):
        image = np.nan_to_num(image).astype(np.uint8)
        # self.setGeometry(self.left, self.top, image.shape[1], image.shape[0])
        #self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        self.q_image = QPixmap.fromImage(self.q_image)
        # self.zoom_height = self.q_image.height()
        # self.image_frame.setPixmap(self.q_image)
        self.viewer.setPhoto(self.image, self.q_image)
        
    def create_thread(self, visualizer):
        self.o3d_vis = visualizer
        thread = QThread()
        self.worker = Action_Poll(parent=self)
        self.worker.moveToThread(thread)
        thread.started.connect(lambda: self.worker.poll_actions(self.o3d_vis))
        # worker.updatedBalance.connect(self.updateBalance)
        self.worker.finished.connect(thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        return thread
    
    def start_threads(self):
        self.threads.clear()
        shapes = []
        for polygon in self.dataset.labels:
            if polygon.view:
                points_3d = utils.get_3d_from_pairs(polygon.points, self.dataset.point_pairs)
                shapes.extend(utils.o3d_polygon(points_3d))

        # utils.display_cloud(self.dataset.cloud, shapes)
        self.threads = [
            self.create_thread(utils.display_cloud(self.dataset.cloud, shapes))
        ]
        for thread in self.threads:
            thread.start()

    def new_visualizer(self):
        self.o3d_viewer.clear_geometries()

    def stop_thread(self):
        self.stop_signal.emit()  # emit the finished signal on stop
        # self.thread.stop()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)


        self.viewer = PhotoViewer(self)
        self.setCentralWidget(self.viewer)

        # self.layout.setAlignment(Qt.AlignTop)
        self.header = QHBoxLayout()
        self.body = QVBoxLayout()
        self.body_horizontal = QHBoxLayout()
        self.body.addLayout(self.body_horizontal)
        self.right_body = QVBoxLayout()

        self.layout.addLayout(self.header)
        self.layout.addLayout(self.body)
        
        
        #self.body_horizontal.addWidget(self.viewer)

        self.tool_buttton_dock = QDockWidget("Tool Types")
        self.tool_bar_widget = QWidget()
        # self.tool_bar_widget.setFixedHeight(90)

        toolbar_layout = QHBoxLayout()
        self.tool_bar_widget.setLayout(toolbar_layout)

        self.tool_toggle_edit = QToolButton()
        self.tool_toggle_edit.clicked.connect(lambda: self.toggle_edit_button())
        self.tool_toggle_edit.setToolTip("Currently Drawing. Click to Edit")
        self.tool_toggle_edit.setIcon(QIcon('icons/color-line.png'))
        self.tool_toggle_edit.setIconSize(QtCore.QSize(50,50))
        self.tool_toggle_edit.setText("ToggleEdit")


        toolbar_layout.addWidget(self.tool_toggle_edit)
        self.tool_buttton_dock.setWidget(self.tool_bar_widget)
        Qt.LeftDockWidgetArea
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tool_buttton_dock)


        self.dock = QDockWidget("File List")
        self.dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        

        # Tool Types
        # LABEL_POLYGON = 0
        # LABEL_LINE = 1
        # LABEL_RECTANGLE = 2
        # LABEL_ELIPSE = 3

        
        # tool_button_polygon = QToolButton()
        # tool_button_polygon.clicked.connect(lambda: self.set_label_mode(LABEL_POLYGON))
        # tool_button_polygon.setToolTip("Polygon Selection Tool")
        # tool_button_polygon.setIcon(QIcon('Polygon_icon.png'))
        # tool_button_polygon.setIconSize(QtCore.QSize(70,70))
        # tool_button_polygon.setText("Polygon Tool")

        # tool_button_line = QToolButton()
        # tool_button_line.clicked.connect(lambda: self.set_label_mode(LABEL_LINE))
        # tool_button_line.setToolTip("Line Selection Tool")
        # tool_button_line.setIcon(QIcon('Line_icon.png'))
        # tool_button_line.setIconSize(QtCore.QSize(70,70))
        # tool_button_line.setText("Line Tool")

        # tool_button_rectangle = QToolButton()
        # tool_button_rectangle.setToolTip("Rectangle Selection Tool")

        # tool_button_elipse = QToolButton()
        # tool_button_elipse.setToolTip("Elispe Selection Tool")

        # toolbar_layout.addWidget(tool_button_polygon)
        # toolbar_layout.addWidget(tool_button_line)
        # toolbar_layout.addWidget(tool_button_rectangle)



    
        self.body_horizontal.addLayout(self.right_body)
        # fileListLayout = QVBoxLayout()
        self.file_list_widget = QListWidget()
        self.file_list_widget.setEnabled(False) # Disables the manual selection of files
        self.file_list_widget.itemDoubleClicked.connect(lambda: self.open_selected_file(False))
        self.dock.setWidget(self.file_list_widget)



        self.label_list_dock = QDockWidget("Label List")
        self.label_list_widget = QListWidget()

        # self.label_list_widget.itemDoubleClicked.connect(self.open_selected_file) #To set the label Info Widget
        self.label_list_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.label_list_dock.setWidget(self.label_list_widget)
        
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_list_dock)

        #LABEL INFO
        self.label_info = QDockWidget("Label Info")
        self.label_info.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        # self.label_container_widget = QWidget()
        self.label_stack_widget = QWidget()
        self.label_stack_layout = QStackedLayout()
        self.label_stack_widget.setLayout(self.label_stack_layout)

        self.label_list_widget.clicked.connect(lambda: self.label_clicked())
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_info)

        # self.label_info_layout = QVBoxLayout()
        # self.label_container_widget.setLayout(self.label_info_layout)
        self.label_info.setWidget(self.label_stack_widget)

        #NOTE PROGRESS BAR WORKS BUT LOOKS WEIRD ON MAC
        # self.load_bar = QProgressBar(self.dock, alignment=Qt.AlignCenter)
        # self.load_bar.setMaximum(100)
        # self.load_bar.setGeometry(60,3,100,15)
        # self.load_bar.setAlignment(Qt.AlignCenter)

        # self.right_body.addWidget(self.load_bar)

        # self.layout.addWidget(self.load_bar)

        self.init_menu()
        self.init_image_UI()



        # self.show_image(self.image)
        # self.add_label() 
        self.show()
    
    def toggle_edit_button(self):
        
        self.edit_mode = not self.edit_mode
        self.edit_toggle.setChecked(self.edit_mode)
        self.viewer.toggle_edit()
        if self.edit_mode:
            self.tool_toggle_edit.setIcon(QIcon("icons/edit.png"))
            self.tool_toggle_edit.setToolTip("Currently Editing. Click to Draw")
        else:
            self.tool_toggle_edit.setIcon(QIcon("icons/color-line.png"))
            self.tool_toggle_edit.setToolTip("Currently Drawing. Click to Edit")

    def set_label_mode(self, mode):
        self.label_mode = mode

    def hide_labels(self):
        '''
        Hides labels on label_list_widget by filenames that exist in dataset
        '''
        for index in range(self.label_list_widget.count()):
            # label = self.dataset.labels[index]
            active_labels = self.dataset.file_labels[self.selected_file]

            print(index, active_labels)

            if index not in active_labels:
                self.label_list_widget.item(index).setHidden(True)
            else:
                self.label_list_widget.item(index).setHidden(False)

            # self.label_stack_layout.setCurrentIndex(index)
            # self.current_polygon = self.dataset.labels[index]

    def populate_labels(self, list):
        '''
        Populates the parent and sibling with dataset labels
        '''
        index = list.currentIndex()
        list.clear()
        list.addItem("None", None)

        polygons = self.dataset.get_polygons_by_file(self.selected_file)
        for polygon in polygons:
            list.addItem(polygon.name, polygon.id)
        
        list.setCurrentIndex(index)

    def populate_siblings(self, unselected_siblings, siblings):

        # Grab all the data in selected siblings and maintain order
        selected = []
        for x in range(siblings.count()):
            selected.append(siblings.item(x).data(1))

        # Clear the two lists
        unselected_siblings.clear()
        siblings.clear()

        # Add selected items into 
        for index in selected:
            item = QListWidgetItem()
            item.setText(self.dataset.labels[index].name)
            item.setData(1, self.dataset.labels[index].id)
            siblings.addItem(item)

        # Create universal set and subtract selected set
        unselected_set = set(range(len(self.dataset.labels)))  - set(selected)
        
        for index in unselected_set:
            item = QListWidgetItem()
            item.setText(self.dataset.labels[index].name)
            item.setData(1, self.dataset.labels[index].id)
            unselected_siblings.addItem(item)



    def label_clicked(self):
        self.hide_labels()

        index = self.label_list_widget.currentRow()
        self.label_stack_layout.setCurrentIndex(index)
        self.label_stack_layout.setEnabled(True)

        # Show selected stack layout from selected label
        widget_stack = self.label_stack_layout.itemAt(self.label_stack_layout.currentIndex()).widget()
        layer_layout = widget_stack.layout()
        
        # Sibling/parent index
        parent_combo = layer_layout.itemAt(5).widget()
        unselected_siblings = layer_layout.itemAt(6).widget().layout().itemAt(0).widget()
        selected_siblings = layer_layout.itemAt(6).widget().layout().itemAt(1).widget()
        self.populate_labels(parent_combo)
        self.populate_siblings(unselected_siblings, selected_siblings)

        widget_stack.setHidden(False) # Unhides the stacked layout

        

        # print("INDEX:", index)
        # self.current_polygon = self.dataset.labels[index]
        self.dataset.current_polygon = self.dataset.labels[index]
        # self.viewer.current_polygon = self.current_polygon
        self.viewer.redraw_all()

    def create_new_label(self):
        #self.label_stack_layout
        self.hide_labels()

        label_info_layout = QVBoxLayout()
        holder_widget = QWidget() # Add this widget dynamically
        holder_widget.setLayout(label_info_layout)

        self.label_stack_layout.addWidget(holder_widget)
        count = self.label_stack_layout.count() -1
        # print("COUNT", count)
        self.label_stack_layout.setCurrentIndex(count)
        # self.label_stack_layout.setCurrentIndex(count)

        label_name = QLineEdit()
        label_name.setPlaceholderText("Label Name")

        label_id = QLineEdit()
        label_id.setPlaceholderText("Label ID")

        distance_label = QLabel()
        distance_label.setText("Distance: 0")

        total_distance_label = QLabel()
        total_distance_label.setText("Distance: 0")

        visible = QCheckBox("Visible")
        visible.setChecked(True)
        # visible.setEnabled(False)
        
        parent_label = QComboBox()
        # parent_label.currentIndexChanged.connect()
        

        relationship_widget = QWidget()
        relationship_layout = QHBoxLayout()

        sibling_priority = QListWidget()
        sibling_priority.setDragDropMode(QAbstractItemView.DragDrop)
        sibling_priority.setFixedWidth(90)
        sibling_priority.setDefaultDropAction(QtCore.Qt.MoveAction)



        # sibling_priority.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        unselected_siblings = QListWidget()
        unselected_siblings.setDragDropMode(QAbstractItemView.DragDrop)
        unselected_siblings.setFixedWidth(90)
        unselected_siblings.setDefaultDropAction(QtCore.Qt.MoveAction)

        relationship_layout.addWidget(unselected_siblings)
        relationship_layout.addWidget(sibling_priority)

        relationship_widget.setLayout(relationship_layout)
        

        delete = QPushButton("Delete")
        delete.clicked.connect(lambda: self.delete_pressed())
        preview = QPushButton("Preview")

        # test = QPushButton("Test")
        # test.clicked.connect(lambda: self.test_poly(self.test_set))

        next = QPushButton("Next")
        next.clicked.connect(lambda: self.next_pressed())

        label_info_layout.addWidget(label_name)
        label_info_layout.addWidget(label_id)
        label_info_layout.addWidget(distance_label)
        label_info_layout.addWidget(total_distance_label)
        label_info_layout.addWidget(visible)
        label_info_layout.addWidget(parent_label)
        label_info_layout.addWidget(relationship_widget)
        label_info_layout.addWidget(delete)
        label_info_layout.addWidget(preview)
        # label_info_layout.addWidget(test)
        label_info_layout.addWidget(next)

        item = QListWidgetItem()
        item.setText(("Label_" + str(count)))
        # item.setCheckState(2)
        # print(count)
    

        self.label_list_widget.addItem(item)
        self.label_list_widget.setCurrentRow(count)
        # self.label_list_widget.item
        # self.label_list_widget.itemSelectionChanged()
        # self.label_list_widget.itemChanged.connect(lambda: self.change_visible(count-1, visible_box=visible))
        
        print(label_name.text())
        label_name.textChanged.connect(lambda: self.handle_label_name_change(label_name, count))
        label_name.textChanged.connect(lambda: self.populate_labels(parent_label))
        label_name.textChanged.connect(lambda: self.populate_siblings(unselected_siblings, sibling_priority))
        parent_label.currentIndexChanged.connect(lambda: self.update_polygon_relationships(parent=parent_label))
        sibling_priority.itemChanged.connect(lambda: self.update_polygon_relationships(siblings=sibling_priority))
        sibling_priority.itemActivated.connect(lambda: self.update_polygon_relationships(siblings=sibling_priority))
        visible.clicked.connect(lambda: self.change_visible(count, item.checkState(), visible))
        preview.clicked.connect(lambda: self.show_labels_3D())

    def next_pressed(self):
        # Test the polygon

        score, passed = self.test_and_record_poly(self.test_set)

        # Score of -1 indicates an error and must fix the error before recording any data.
        if score == -1:
            return
        # self.scores.add_polyscore(self.selected_file, self.dataset.current_polygon.total_distance, passed, self.scores.training_phase)

        msg = QMessageBox()
        msg.setWindowTitle("Results")

        # Only changes to testing phase when training has been completed.
        if self.scores.passed_training() and self.scores.training_phase:
            text = ("Score:" + str(self.scores.training_score) + "Press OK to continue. \nYou are now entering the Testing Phase")
            msg.setText(text)
            x = msg.exec_()  # this will show our messagebox

            self.scores.training_phase = False
            testing_folder = os.path.expanduser(os.getcwd()) + "/data/Testing/"
            self.get_folder(testing_folder)
            self.load_random(reload_order=True)


        elif not self.scores.passed_training():
            text = ("Score:" + str(self.scores.correct_train) + "/" + str(self.scores.training_pass_limit) + " : " + str(self.scores.training_score*100) + "Press OK to continue.")
            msg.setText(text)
            x = msg.exec_()  # this will show our messagebox

            self.load_random(False)

        elif self.scores.number_tested >= self.scores.total_test_files and self.scores.total_test_files != 0:
            text = ("Score:" + str(self.scores.testing_score) + "\nYou are complete. Please submit your results.")
            msg.setText(text)
            x = msg.exec_()  # this will show our messagebox

        else:
            self.load_random(False)
        print(self.scores.number_tested, self.scores.total_test_files)

        

    def get_current_info_widgets(self, item_at):
        # layout = self.label_stack_layout.children()[0].children()[0].children()
        # print(self.label_stack_layout.currentIndex())
        # if self.label_stack_layout.currentIndex() >= 0:
        holder = self.label_stack_layout.itemAt(self.label_stack_layout.currentIndex()).widget() # We get Holder Widget
        layout = holder.layout()
        item = layout.itemAt(item_at).widget()
        return item
    
    def get_info_widget(self, index ,item_at):
        holder = self.label_stack_layout.itemAt(index).widget() # We get Holder Widget
        layout = holder.layout()
        return layout.itemAt(item_at).widget()
        
    def handle_label_name_change(self, name_widget, label_index):
        print("SELECTED")
        new_text = name_widget.text()
        self.dataset.current_polygon.set_label(new_text)
        try:
            self.label_list_widget.item(label_index).setText(new_text)
        except Exception as e:
            print(e)
            print("Could not change name on item that does not exist")
        self.viewer.redraw_all()

    def update_polygon_relationships(self, parent=None, siblings=None):
            
        if parent:
            selected_id = parent.currentData(1) # Grabs the polygon id
            self.dataset.current_polygon.parent = self.dataset.get_polygon_by_id(selected_id)
        
        if siblings:
            
            # Create sibling list before we assign it
            sibling_list = []
            for index in range(siblings.count()):
                sibling_id = siblings.item(index).data(1) # data(1) is the data location for polygon id
                sibling_polygon = self.dataset.get_polygon_by_id(sibling_id)
                sibling_list.append(sibling_polygon)

            

            # for sibling_polygon in sibling_list:
            #     sibling_polygon.siblings = sibling_list

    def hide_all(self):
        for index in range(self.label_list_widget.count()):
            self.get_info_widget(index, 4).setCheckState(False)
            
        for label in self.dataset.labels:
            label.view = False
        
    def show_selected_file(self):
        label_indexes = self.dataset.file_labels[self.selected_file]
        for index in label_indexes:
            label = self.dataset.get_polygon(index)
            label.view = True
            print("Label index", index)
            self.get_info_widget(index, 4).setCheckState(2)

        # self.file_list_widget.selectedItems()[0].text()


    def change_visible(self, index, set_value=None, visible_box=None):
        self.dataset.labels[index].view = not self.dataset.labels[index].view
        self.viewer.redraw_all()
        
    def show_labels_3D(self, index=None):
        '''
        '''
        if self.o3d_vis is None:
            self.start_threads()

    def delete_pressed(self):
        # Get active labels in file and current index in list
        current_index = self.label_stack_layout.currentIndex()
        active_labels = self.dataset.file_labels[self.selected_file]
        print("Current Index", current_index)
        print("Active Label Length", len(active_labels))

        # Get the index of current label in the active_labels list.
        active_index = active_labels.index(current_index)

        holder = self.label_stack_layout.itemAt(current_index).widget() # We get Holder Widget
        layout = holder.layout()
        holder.setParent(None)
        # Delete the polygon in dataset

        # Determine the next layout and label to activate
        next_index = -1
        # if list is larger than 1, we change the index
        if len(active_labels) > 1:
            # If index is the first in the list  set it to the next
            if active_index == 0:
                next_index = active_index + 1
            
            # If the label is the last of the active labels, set it to the second last
            elif active_index == len(active_labels) - 1:
                
                next_index = active_index - 1


        print("Next Index", next_index)

        if next_index >= 0:
            # Update the stack layout with the information the the next indexed polygon
            self.label_stack_layout.setCurrentIndex(next_index)
            self.label_list_widget.setCurrentRow(next_index)
        
        else:
            if self.label_stack_layout.count() > 0:
                widget = holder = self.label_stack_layout.itemAt(self.label_stack_layout.currentIndex()).widget()
                widget.setHidden(True)

            
        # Do this reguardless what the next item is
        self.dataset.delete_polygon(current_index, next_index=next_index)
        #Update viewer polygon to dataset's polygon
        # self.viewer.current_polygon = self.dataset.current_polygon
        # self.current_polygon = self.dataset.current_polygon

        # Hide deleted item
        self.label_list_widget.setRowHidden(current_index, True)
        item = self.label_list_widget.takeItem(current_index)
        del item

    def init_image_UI(self):
        pass

    def init_file_UI(self):
        pass

    def init_buttons(self):
        self.drag_start = False
    
    def init_menu(self):
        bar = QMenuBar()
        bar.setMaximumHeight(20)
        
        self.header.addWidget(bar)
        
        file = bar.addMenu("File")

        open_file = QAction("Open File", self)

        open_folder = QAction("Open Folder", self)
        open_folder.triggered.connect(lambda: self.get_folder())
        

        load_csv = QAction("Load from CSV", self)
        load_csv.triggered.connect(lambda: self.load_labels())

        load_training_dataset = QAction("Load training dataset", self)
        load_training_dataset.triggered.connect(lambda: self.load_labels(show_ui=False, dataset=self.training_datast))


        load_testing_dataset = QAction("Load testing dataset", self)
        load_testing_dataset.triggered.connect(lambda: self.load_labels(show_ui=False, dataset=self.testing_dataset))
        # export_all.triggered.connect(lambda: pass))

        export_all = QAction("Export All", self)
        export_all.triggered.connect(lambda: self.dataset.export_all_labels(self.user, self.current_folder))

        # export_coco = QAction("Save Coco Labels", self)
        # export_coco.triggered.connect(lambda: self.dataset.save_coco_file(self.selected_file, self.coco_labels, self.original_img))

        export_json = QAction("Save ply as Json", self)
        export_json.triggered.connect(lambda: self.dataset.export_ply_and_labels(self.dataset.current_file))

        autosave = QAction("Autosave", self, checkable=True)
        autosave.setChecked(True)
        autosave.triggered.connect(lambda: self.toggle_autosave())
        
        file.addAction(open_file)
        file.addAction(open_folder)
        file.addAction(load_csv)
        file.addAction(load_training_dataset)
        file.addAction(load_testing_dataset)
        # file.addAction(export_coco)
        file.addAction(export_all)
        file.addAction(export_json)

        file.addAction(autosave)

        edit = bar.addMenu("Edit")

        edit_undo = QAction("Undo", self)
        edit_undo.triggered.connect(lambda: self.undo())
        edit_undo.setShortcuts([QtGui.QKeySequence(Qt.CTRL + Qt.Key_Z)])

        edit_redo = QAction("Redo", self)
        edit_redo.triggered.connect(lambda: self.redo())
        edit_redo.setShortcuts([QtGui.QKeySequence(Qt.CTRL + Qt.Key_Y)])

        self.edit_toggle = QAction("Toggle Edit Mode", self, checkable=True)
        self.edit_toggle.setChecked(False)
        self.edit_toggle.triggered.connect(lambda: self.toggle_edit_button())
        self.edit_toggle.setShortcuts([QtGui.QKeySequence(Qt.Key_E)])

        edit_new_label = QAction("New Label", self)
        edit_new_label.triggered.connect(lambda: self.add_label())
        edit_new_label.setShortcuts([QtGui.QKeySequence(Qt.Key_Space)])

        edit_remove_point = QAction("Remove Point", self)
        edit_remove_point.triggered.connect(lambda: self.dataset.remove_point(self.viewer))
        edit_remove_point.setShortcuts([QtGui.QKeySequence(Qt.Key_Backspace)])


        edit.addAction(edit_undo)
        edit.addAction(edit_redo)
        edit.addAction(self.edit_toggle)
        edit.addAction(edit_new_label)
        edit.addAction(edit_remove_point)

        self.setMenuBar(bar)


        view = bar.addMenu("View")

        increase_cursor = QAction("Incerase Cursor", self)
        increase_cursor.triggered.connect(lambda: self.viewer.increase_cursor())
        # increase_cursor.setShortcuts([QtGui.QKeySequence(Qt.Key_Plus)])

        decrease_cursor = QAction("Decrease", self)
        decrease_cursor.triggered.connect(lambda: self.viewer.decrease_cursor())
        # decrease_cursor.setShortcuts([QtGui.QKeySequence(Qt.Key_Minus)])
        

        zoom_in = QAction("Zoom In", self)
        zoom_in.triggered.connect(lambda: self.viewer.zoom_in())
        zoom_in.setShortcuts([QtGui.QKeySequence(Qt.Key_Plus)])
        


        zoom_out = QAction("Zoom Out", self)
        zoom_out.triggered.connect(lambda: self.viewer.zoom_out())
        zoom_out.setShortcuts([QtGui.QKeySequence(Qt.Key_Minus)])

        view.addAction(zoom_in)
        view.addAction(zoom_out)
        

        view.addAction(increase_cursor)
        view.addAction(decrease_cursor)

        help = bar.addMenu("Help")

        wiki = QAction("Help Page", self)
        wiki.triggered.connect(lambda: webbrowser.open('https://github.com/hobbitsyfeet/RemoteAnimalScan/wiki/Buddy-Measure-User-Guide'))

        help.addAction(wiki)


        dev = bar.addMenu("Developer")
        access_file_list = QAction("Access File List", self, checkable=True)
        access_file_list.setChecked(False)
        access_file_list.triggered.connect(lambda: self.file_list_widget.setEnabled(access_file_list.isChecked()))

        dev.addAction(access_file_list)

    def undo(self):  
        self.dataset.undo()
        self.viewer.redraw_all()
    
    def redo(self):
        self.dataset.redo()
        self.viewer.redraw_all()

    def toggle_autosave(self):
        self.autosave = not self.autosave
    
    def load_labels(self, show_ui=True, dataset=None, path=None):
        "Loads points onto filename"

        # temperary_selected = 
        if dataset is None:
            dataset = self.dataset
        
        if path is None:
            path = str(QFileDialog.getOpenFileName(self, "Select CSV to Load Labels", )[0])

        csv = pd.read_csv(path)
        # Iterate through each row. One row is one polygon.
        for index, row in csv.iterrows():
            target_file = row["Filename"]
            # print("loading labels for file:", target_file)
            # self.selected_file = target_file
            points = row["Points"]

            # Parse Points from text to list of tuples
            points = eval(points)

            # Create polygon at target_filename
            label_index = dataset.create_polygon(target_file)

            # Apply points to appropriate filename
            dataset.assign_polygon(label_index, points)
            
            if show_ui:
                # Covers the UI portion of the label
                self.create_new_label()
                self.show_selected_file()
            else:
                # Hide the label
                dataset.labels[label_index].view = False

            # Creating mode false (closes polygon)
            dataset.labels[label_index].creating = False

        print("Labels loaded.")

        if self.scores.training_phase:
            self.scores.total_train_files = len(self.file_list)
        else:
            self.scores.total_test_files = len(self.file_list)

    def add_label(self):
        print("Creating label")
        if self.dataset.current_polygon is not None:
            self.dataset.current_polygon.creating = False
        # self.dataset.create_polygon(utils.get_global_filename(self.current_folder, self.file_list[0]))
        print(self.selected_file)
        index = self.dataset.create_polygon(self.selected_file)
        self.create_new_label()
        self.hide_labels()
        # self.viewer.set_polygon(self.dataset.current_polygon)

    def map_to_widget(self, widget, point):
        # gp = widget.mapToGlobal(QtCore.QPoint(0, 0))
        # b_a = widget.mapFromGlobal(gp)
        # print()
        # # print(b_a.x(),b_a.y())
        mapped = ((point[0] - widget.pos().x()), point[1] - widget.pos().y())

        return mapped

    def toggle_edit(self):
        self.edit_mode = not self.edit_mode

    def open_selected_file(self, already_selected = False):
        '''
        The main function which loads selected_file into the active screen to label.
        '''
        # file = self.file_list_widget.item(0).text()
        
        # Check if polygons have not been saved, if not, save them
        print("DATASET IS EMPTY?: ", self.dataset.empty())
        if self.autosave and not self.dataset.empty():
            self.dataset.export_all_labels(self.user, self.current_folder, use_previous_path=True)
        
        if already_selected is False:

            self.selected_file = self.file_list_widget.selectedItems()[0].text()
        # self.dataset.current_file = self.selected_file
        print(self.selected_file)
        extention = self.selected_file.split('.')[-1]

        if extention == "ply" or extention == "projected" or extention =="json":
            
            self.hide_all()
            self.dataset.load_cloud(utils.get_global_filename(self.current_folder, self.selected_file))
            self.image = self.dataset.image
            
            # print(self.dataset.file_labels.keys())
            
            if self.selected_file in self.dataset.file_labels.keys():
                self.show_selected_file()
                #set the current index to the first label in the list
                active_labels = self.dataset.file_labels[self.selected_file]
                # add a label if it is empty to prevent a crash
                if len(active_labels) == 0:
                    self.add_label()

                # print(active_labels[0])
                self.label_list_widget.setCurrentRow(active_labels[0])
                # self.label_stack_layout.setCurrentIndex(active_labels[0]+1)

            else:
                self.add_label()
            

            # self.viewer.update_image(self.image)
        else:
            self.image = cv2.imread(utils.get_global_filename(self.current_folder, self.selected_file))
        
        self.original_img = deepcopy(self.image)
        
        # We set viewer image to none before setting a new one because none is a case which determines a new original copy
        self.viewer.image = None
        self.viewer.setPhoto(self.image)
        self.viewer.fitInView()
        self.viewer.redraw_all()
        self.label_clicked()
        self.viewer.update_o3d_viewer()
        # if self.o3d_vis is not None:
            
        
        # self.image = self.viewer.draw_polygons(self.image)
        # self.viewer.update_image(self.image)

        # file = str(QFileDialog.getOpenFileName(self, "Select Directory")[0])
    
    def get_global_filename(self, folder, filename):
        print(str((folder + "/" + filename)))
        return str((folder + "/" + filename))

    def get_filename_index(self, filename):
        for index, item in self.file_list_widget.items():
            if str(item) == filename:
                return index
        return None

    def get_file_by_index(self, file_index):
        for index, item in self.file_list_widget.items():
            if file_index == index:
                return item
        return None

    def get_folder(self, foldername = None):
        if foldername is None:
            self.current_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        else:
            self.current_folder = foldername
        print(self.current_folder)
        self.file_list = os.listdir(self.current_folder)
        # for file in folder:
        # self.folder_explorer.setText(folder)
        self.file_list_widget.clear()
        self.file_list_widget.addItems(self.file_list)
        self.file_list_widget.repaint()
        self.file_list_widget.show()

        self.load_random(True, self.user)
        
        print("Loaded",  self.file_list)

    def load_random(self, reload_order=True, seed=None):
        '''
        Load random will take a random file in the testing set and load it for the user.

        Reload order will resample the file list in the folder, randomize it and 
        start on the first of the list
        '''
        # file_list = self.file_list_widget.items()
        # self.file_list
        print("RANDOM LOAD FILE LIST", self.file_list)
        if self.load_order is None or reload_order:

            if seed is not None:
                random.seed(seed)

            self.load_order = random.sample(self.file_list, k=len(self.file_list))
            self.load_order_index = 0

        # Select the file before we go to the next random image
        self.selected_file = self.load_order[self.load_order_index]
        print("SELECTED FILE", self.selected_file)

        # Increment load order index and modulo the number within the range. 
        # This makes the list repeat itself when next is above the range
        self.load_order_index = (self.load_order_index + 1) % len(self.load_order)
        print("Load Order" , self.load_order_index, self.selected_file, '\n',  self.load_order)


        self.open_selected_file(already_selected=True)


    def test_and_record_poly(self, dataset, Training=True):
        '''
        Tests if both points fall within a testing polygon
        '''
        if dataset.empty():
            print("Training dataset is empty. Load a dataset and try again.")
            return 

        print("TESTED SELECTED FILE", self.selected_file)
        testing_polygons = self.dataset.get_polygons_by_file(self.selected_file)
        print("TESTING POLYGONS", testing_polygons)
        # Make sure at least 2 polygons and every polygon has at least 2 points

        if len(testing_polygons) < 2:
            msg = QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("You need at least 2 polygons.")
            x = msg.exec_()  # this will show our messagebox
            return -1, False

        for polygon in testing_polygons:   
            if len(polygon.points) < 2:
                msg = QMessageBox()
                msg.setWindowTitle("Error")
                msg.setText("Every polygon needs at least 2 points.")
                x = msg.exec_()  # this will show our messagebox
                return -1, False


        fit_count = 0
        passed = False
        train_pass = False
        both_test = False
        smallest_index = 0

        # Get polygon keys to test against user training polygons
        train_keys = dataset.file_labels[self.selected_file]      

        # Test all the polygons in current dataset
        for test_polygon in testing_polygons:
            # reset fit count
            fit_count = 0
            train_pass = False

            # There should be 2 ground truth polygons
            for index, key in enumerate(train_keys):
                # if self.dataset.get_filename(test_polygon) == self.selected_file:
                train_polygon = dataset.get_polygon(key)

                image = train_polygon.draw(self.viewer.image, overwrite_label=" ", show_points=False, infill=(255,255,204,0.5), line_color=(0,0,0), thickness=1)
                self.viewer.update_image(image)
                
                fit = self.dataset.measure_polygon_fit(test_polygon, train_polygon)

                # Test if 2 points exist inside of designated polygons
                if len(train_polygon.points) > 2:
                    if fit:
                        fit_count += 1

                    # else:
                    #     image = train_polygon.draw(self.viewer.image, overwrite_label=" ", infill=(0,0,0,0.2), line_color=(255,0,0), thickness=1)
                    #     self.viewer.update_image(image)

                if fit_count == 2:
                    train_pass = True
                    passed = True
                # Record the score
                self.scores.add_polyscore(self.selected_file, test_polygon, passed=passed, Training=Training)
            print("TRAIN PASS", train_pass, polygon.points)
            # print("FIT COUNT", fit_count)

            if train_pass:
                image = test_polygon.draw(self.viewer.image, overwrite_label=" ", line_color=(0,255,0), thickness=2)
                self.viewer.update_image(image)
            else:
                image = test_polygon.draw(self.viewer.image, overwrite_label=" X ", line_color=(0,0,255), thickness=2)
                self.viewer.update_image(image)



        return fit_count, passed

    def testing_phase_check(self):
        if self.scores.training_phase and self.scores.passed_training():
            # Exit training phase (go to testing phase)
            self.scores.training_phase = False

            # Load testing folder and 
            testing_folder = os.path.expanduser(os.getcwd()) + "/data/Testing/"
            self.get_folder(testing_folder)
            # self.load_random(reload_order=True)

        # If in the testing phase and file limit has been reached, you're done
        if self.scores.training_phase == False and self.scores.number_tested == self.scores.testing_file_number:
            # This is to pause to show result before moving on
                msg = QMessageBox()
                msg.setWindowTitle("Results")
                msg.setText("You have completed the testing set. Please submit the results.")
                x = msg.exec_()  # this will show our messagebox

    def test_training_line(self, upperlimit=10):
        '''
        Tests against training set based on a list of polygons. 
        This tests a score based on line similarity with maximum real-world euclidean distance in m. 
        We define a 10mm (1cm) maximum error between a set of 2 points
        '''
        if self.training_datast.empty():
            print("Training dataset is empty. Load a dataset and try again.")
            return
        # Get current polygon
        test_polygon = self.dataset.current_polygon

        if len(test_polygon.points) < 2:
            print("You must assign a polygon with 2 points to test first.")
            return
        scores = []
        score_polygon = []

        smallest_index = 0
        # Get polygon from the same file
        for train_file in self.training_datast.file_labels.keys():
            train_keys = self.training_datast.file_labels[train_file]

            # iterate becasuse it is a list, though it might only have one
            for index, key in enumerate(train_keys):
                if self.dataset.get_filename(test_polygon) == train_file:
                    train_polygon = self.training_datast.get_polygon(key)

                    # Test the difference
                    score = self.dataset.measure_line_similarity(test_polygon, train_polygon, upperlimit)
                    scores.append(score)

                    

                    if score < scores[smallest_index]:
                        smallest_index = index

                    score_polygon.append(train_polygon)


        

        print(scores)
        
        if scores[smallest_index] < 1:
            image = score_polygon[smallest_index].draw(self.viewer.image, overwrite_label="Correct!", line_color=(0,255,0), thickness=2)
            self.viewer.update_image(image)

            # Record passed score
            
            
        else:
            image = score_polygon[smallest_index].draw(self.viewer.image, overwrite_label="X", line_color=(0,0,255), thickness=2)
            self.viewer.update_image(image)

                




'''
######################################################################
CLASS PHOTOVIEWER


######################################################################
'''
class PhotoViewer(QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self.parent = parent
        # self.dataset = dataset
        self.image = None
        self.original_img = None
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self._painter = QtGui.QPainter()
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

        self.hover_point = None
        self.drag_start = False
        self.edit_mode = False
        self.mouse_pos = None
        self.selected_point = None
        # self.current_polygon = None
        self.cursor_size = 1

        self.setDragMode(QGraphicsView.NoDrag)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # self.update_visualizer()


    def dataset(self):
        """
        Returns the parent's dataset
        """
        return self.parent.dataset

    def start():
        pass

    def increase_cursor(self):
        self.cursor_size += 1
    
    def decrease_cursor(self):
        self.cursor_size -= 1

    @QtCore.pyqtSlot()
    def update_image(self, image):
        '''
        This assignes a numpy image to Qimage and Pixmap.

        Use when done adding elements to the image. (Does not provide clean image, only assignes image passed in to viewer)

        NOTE: Use redraw_image when wanting to work with a clean slate.
        '''
        image = np.nan_to_num(image).astype(np.uint8)
        # self.setGeometry(self.left, self.top, image.shape[1], image.shape[0])
        self.q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.q_image = QPixmap.fromImage(self.q_image)
        # self.zoom_height = self.q_image.height()
        # self.image_frame.setPixmap(self.q_image)
        self.setPhoto(image, self.q_image)

    def draw_polygons(self, image):
        for polygon in self.dataset().labels:
            if polygon.view is True:
                image = polygon.draw(image)
        return image

    def toggle_edit(self):
        self.edit_mode = not self.edit_mode

    # def set_polygon(self, current_polygon):
        # self.current_polygon = current_polygon

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    @QtCore.pyqtSlot()
    def redraw_image(self):
        '''
        Resets to a copy of the original image (clean slate)

        Used when drawing so objects are not redrawn infinitely. I.E mouse location
        '''
        self.image = deepcopy(self.original_img)

    def redraw_all(self):
        self.redraw_image()
        self.update_image(self.draw_polygons(self.image))
        self.update_o3d_viewer()

    def setPhoto(self, image, pixmap=None):
        if self.image is None:
            self.image = image
            self.original_img = deepcopy(self.image)
        else:
            self.image = image
            
            
        # print(image)
        image = np.nan_to_num(image).astype(np.uint8)
        pixmap = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(pixmap)

        # self._zoom = 100
        if pixmap and not pixmap.isNull():
            self._empty = False
            # self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        

    def wheelEvent(self, event):
        if self.image is None:
            return

        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):

        if self.image is None:
            return

        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
            # print(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

        self.drag_start = True
        print("drag start")
        # mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))
        
        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())
        self.mouse_pos = mapped_point
        # self.map_to_widget(self.image_frame, point)
        
        if self.dataset().current_polygon is None:
            # print(self.parent.selected_file)
            self.parent.add_label()
            # self.dataset.create_polygon(self.parent.selected_file)
            # self.current_polygon = self.dataset.current_polygon

        if self.dataset().current_polygon is not None:
            
            # If we have a polygon, save the new state 
            if self.dataset().current_polygon.queue_save is True:
                self.parent.dataset.save_state()
                self.dataset().current_polygon.queue_save = False


            # Assign point
            if event.button() == Qt.LeftButton and not self.edit_mode:
                print("Left Button Clicked")
                self.dataset().save_state()
                self.dataset().current_polygon.assign_point(mapped_point)
                self.redraw_image()
                image = self.dataset().current_polygon.draw(self.image)
                
                self.update_image(image)
                self.update_distance(mapped_point)
                
                print(self.parent.o3d_vis)
                if self.parent.o3d_vis is not None:
                    print("Visualizer", self.parent.o3d_vis)
                    print("Updating visualizer")
                    
                    points_3d = utils.get_3d_from_pairs(self.dataset().current_polygon.points, self.dataset().point_pairs)
                    poly = utils.o3d_polygon(points_3d)
                    utils.o3d_add_object(self.parent.o3d_vis, poly)
                    self.update_o3d_viewer()
                    # self.parent.o3d_vis.update_geometry(poly)
                    # self.parent.o3d_vis.poll_events()
                    # self.parent.o3d_vis.update_renderer()
                    
            

            if event.button() == Qt.RightButton:

                # Save line and test against training set
                # print("Right button clicked")
                # self.redraw_image()
                # self.dataset().save_state()
                # point = self.dataset().current_polygon.check_near_point(mapped_point, dist=2)
                # self.dataset().current_polygon.remove_point(point)
                # self.selected_point = None
                
                # image = self.dataset().current_polygon.draw(self.image)
                # self.update_image(image)
                


                # '''
                #NOTE this comment block cuts segment and displays it in 3D
                self.dataset().current_polygon.end_poly()
                self.redraw_image()
                image = self.dataset().current_polygon.draw(self.image)
                mask = self.dataset().current_polygon.create_mask(self.image.shape[0], self.image.shape[1])
                self.dataset().save_state()

                # cropped , coordinates, cloud = self.current_polygon.get_segment_crop(self.original_img, self.parent.dataset.point_pairs)
                # '''
                # self.update_image(image)
            
            #Edit mode select point
            if event.buttons() == QtCore.Qt.LeftButton and self.edit_mode:
                
                #Distance for selected point (should match mouse move event and change colour to indicate accurate selection)
                point = self.dataset().current_polygon.check_near_point(mapped_point, dist=2)
                self.selected_point = point
                

                # self.selected_point = poin

                # self.current_polygon = None
                
    # def paintEvent(self, event):
    # #     print("Paint event")
    #     if self.mouse_pos is not None:
    #         print("painting")
    #         self.draw_points()

    # @QtCore.pyqtSlot()
    # def draw_points(self):
    #     if self.image is None:
    #         return

    #     if self.drag_start:
    #         self._painter.begin(self._scene)
            
    #         self._painter.setRenderHint(QPainter.Antialiasing)
    #         pen = QPen(Qt.red, 7)
    #         brush = QBrush(Qt.red)
    #         self._painter.setPen(pen)
    #         self._painter.setBrush(brush)
    #         point = QtCore.QPoint(self.mouse_pos[0], self.mouse_pos[1])
    #         self._painter.drawPoint(point)
    #         # self._painter.drawPixmap(point)
    #         self._painter.end()
    def zoom_in(self, ammount=1):
        factor = 1.25
        self._zoom += ammount
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    def zoom_out(self, ammount=1):
        factor = 0.8
        self._zoom -= ammount
        if self._zoom > 0:
            self.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0

    def mouseReleaseEvent(self, event):
        
        if self.image is None:
            return

        # mapped_point = self.map_to_widget(self.image_frame, (event.x(), event.y()))
        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())

        self.drag_start = False
        print("dropped")



        if self.dataset().current_polygon is not None:

            if self.dataset().current_polygon.queue_save is True:
                self.dataset().save_state()
                self.dataset().current_polygon.queue_save = False

            if self.selected_point is not None and self.edit_mode:
                self.redraw_image()
                self.dataset().save_state()
                self.dataset().current_polygon.edit_point(self.selected_point, mapped_point)

            # if self.parent.o3d_vis is not None:
            self.update_o3d_viewer()

            # image = self.current_polygon.draw(self.image)
            image = self.draw_polygons(self.image)
            self.update_image(image)
            self.selected_point = None

        labels = self.dataset().labels
        # for polygon1 in labels:
        #     for polygon2 in labels:
        #         if polygon1 != polygon2:
        #             score = self.dataset().measure_line_similarity(polygon1, polygon2, upper_limit=20)
                    # print(polygon1.name, polygon2.name, score)
                

    def update_o3d_viewer(self):
        if self.parent.o3d_vis is not None:
            cleard = self.parent.o3d_vis.clear_geometries()
            # print("Cleared: ", cleard)
            utils.o3d_add_object(self.parent.o3d_vis, [self.parent.dataset.cloud])

            if self.dataset().current_polygon:
                points_3d = utils.get_3d_from_pairs(self.dataset().current_polygon.points, self.dataset().point_pairs)
                poly = utils.o3d_polygon(points_3d)
                utils.o3d_add_object(self.parent.o3d_vis, poly)

            # Cursor location
            pointer_3d = utils.get_3d_from_pairs([self.mouse_pos], self.dataset().point_pairs)
            if len(pointer_3d) >= 1:
                pointer_box = utils.o3d_box(pointer_3d[0])
                utils.o3d_add_object(self.parent.o3d_vis, [pointer_box])


    @QtCore.pyqtSlot()
    def mouseMoveEvent(self, event):

        if self.image is None:
            return

        mapped_point = self.mapToScene(event.pos()).toPoint()
        mapped_point = (mapped_point.x(), mapped_point.y())
        self.mouse_pos = mapped_point

        self.update_distance(mapped_point)

        if self.parent.o3d_vis is not None:
            self.update_o3d_viewer()
        # try:
        if self.parent.worker is not None:
            if self.parent.worker.continue_run is False:
                if self.parent.o3d_vis is not None:
                    self.parent.o3d_vis.destroy_window()
        # except:
            # pass

        try:
            image = np.nan_to_num(self.original_img).astype(np.uint8)
            b,g,r = image[mapped_point[1]][mapped_point[0]]

            self.redraw_image()
            image = cv2.circle(image, mapped_point, radius=self.cursor_size, color=(int(255-b),int(255-g),int(255-r)), thickness=1)

            if mapped_point in self.dataset().point_pairs.keys():
                point, colour = self.dataset().point_pairs[mapped_point]
                # print(point[2])
                image = cv2.putText(image, text=(str((point[2]/1000)) + "m"), org=mapped_point, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.75, color=(int(255-b),int(255-g),int(255-r)),thickness=1, lineType=cv2.LINE_AA)
            # image = cv2.rectangle(self.image, mapped_point, mapped_point, (255-b,255-g,255-r))
            # image = self.current_polygon.draw(image)
            image = self.draw_polygons(image)
            # image = cv2.circle(self.image, self.hover_point, 1, (0,255,255), 1)
            self.update_image(image)
        except IndexError as e:
            print(e)
        except:
            # print(e)
            print("Could not grab mapped point.")

        if event.buttons() == QtCore.Qt.NoButton:

            if self.dataset().current_polygon is not None:
                #Check when hovering is within distance value to change colour
                point = self.dataset().current_polygon.check_near_point(mapped_point, dist=2)

                #When hovering over point
                if point is not None:
                    image = cv2.circle(self.image, point, self.cursor_size, (0,255,0), 1)
                    # image = cv2.circle(self.image, point, 5, (0,0,0), 1)
                    self.hover_point = point
                    self.update_image(image)

                #When hover point is left alone
                elif self.hover_point is not None and point is None:
                    # self.redraw_image()
                    
                    image = cv2.circle(self.image, self.hover_point, self.cursor_size, (0,255,255), 1)
                    # image = cv2.circle(self.image, self.hover_point, 4, (0,0,0), 2)
                    self.hover_point = None

                    self.update_image(image)


        if self.drag_start and event.buttons() != Qt.RightButton:

            print("dragging")
            # if event.button() == Qt.LeftButton:
            if self.dataset().current_polygon is not None:

                if self.dataset().current_polygon.queue_save is True:
                    self.dataset().save_state()
                    self.dataset().current_polygon.queue_save = False

                #Check if hover is within distance for assigning a point (distance between new points, used for hold and drag assignment)
                point = self.dataset().current_polygon.check_near_point(mapped_point,dist=5)

                if point is None and not self.edit_mode:
                    self.redraw_image()
                    self.dataset().current_polygon.assign_point(mapped_point)
                    image = self.dataset().current_polygon.draw(self.image)
                    self.update_image(image)

            #If edit mode and point is selected
            if self.edit_mode is True and self.selected_point is not None:
                adjacent_points = self.dataset().current_polygon.get_adjacent(self.selected_point)
                #if there is an adjacent point
                if adjacent_points[0] is not None or adjacent_points[1] is not None:
                    self.redraw_image()
                    image = self.dataset().current_polygon.draw(self.image)
                    if adjacent_points[0] is not None:
                        image = cv2.line(image, adjacent_points[0], mapped_point, (0,255,255), 1)
                    if adjacent_points[1]is not None:
                        image = cv2.line(image, adjacent_points[1], mapped_point, (0,255,255), 1)
                    self.update_image(image)

            # update distance
            self.update_distance(mapped_point, editing=True)

            # # update 3D view if available
            # if self.parent.o3d_vis is not None:
            #     self.update_o3d_viewer()
                

        # else:
            # image = cv2.circle(self.image, self.hover_point, 4, (0,0,0), 2)
            # self.hover_point = None

    def update_distance(self, mapped_point, editing=False):
        # print("updating distance")
        if self.dataset().current_polygon is not None:
            if len(self.dataset().current_polygon.points) >= 1:
                
                total_distance = utils.get_total_distance(self.dataset().point_pairs, self.dataset().current_polygon.points)
                self.dataset().current_polygon.total_distance = total_distance/10
                # print(total_distance)
                self.parent.get_current_info_widgets(3).setText(("Total Distance: " + str(total_distance/10) + "cm"))       

                # Measure the distance between points
                p2 = mapped_point
                p1 = self.dataset().current_polygon.points[-1]
        
                # if editing and len(self.current_polygon.points) >= 2:
                    # p2 = mapped_point
                    # p1 = self.current_polygon.points[-1]
                # elif len(self.current_polygon.points) >= 2:
                #     p1 = self.current_polygon.points[-2]
                #     p2 = self.current_polygon.points[-1]

                if p1 in self.dataset().point_pairs.keys() and p2 in self.dataset().point_pairs.keys():
                    distance = utils.get_distance_2D(self.dataset().point_pairs, p1, p2)
                    self.parent.get_current_info_widgets(2).setText(("Distance: " + str(distance/10) + "cm"))



def cloud_to_image():
    pass

def image_to_cloud():
    pass

def display_2D():
    pass

def display_3d():
    pass


def label_polygon():
    pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())