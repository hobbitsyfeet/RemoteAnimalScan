"""
Handheld Records data when using a mobile device
    Requirements: Nvidia Jetson (Nano or other) ARM CPU 64-bit
    Optional: Button trigger, Screen, LCD (Liquid-Crystal Display)
"""
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QIntValidator, QPixmap, QImage
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                             QComboBox, QDockWidget, QDoubleSpinBox,
                             QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSpinBox, QStackedLayout, QStyle,
                             QTextEdit, QVBoxLayout, QWidget, QMenuBar)
import sys
import cv2
import Record



class Recorder(QWidget):
    def __init__(self):
        super().__init__()

        self.camera_type = None
        self.ffmpeg_path = "C:/Users/legom/Downloads/ffmpeg-20200729-cbb6ba2-win64-static/ffmpeg-20200729-cbb6ba2-win64-static/bin/ffmpeg.exe"
        self.init_Window()
        camera, camera_type = self.setup_cameras()
        print(camera)

        self.show()
        while True:
            colour, depth = self.read(camera, camera_type)
            self.view_2d(colour)
            QApplication.processEvents()


    def init_Window(self):
        vbox = QVBoxLayout()
        header = QHBoxLayout()
        body = QHBoxLayout()
        vbox.addLayout(header)
        vbox.addLayout(body)

        self.init_viewer(body)
        self.setLayout(vbox)
        self.setGeometry(500, 500, 500, 400)
        self.setWindowTitle('Recorder')
        self.setWindowIcon(QIcon('web.png'))

    def init_viewer(self, layout):
        viewer_layout = QHBoxLayout()
        layout.addLayout(viewer_layout)

        self.view_label = QLabel()
        self.view_label.resize(600, 600)
        
        viewer_layout.addWidget(self.view_label)
    
    def record_ply(self):
        pass

    def get_intrinsics(self):
        pass
    
    def record_colour(self, filename, frame_rgb, fps=30, frame_size = (720,480)):
        out = cv2.VideoWriter(filename + "_RGB.avi", -1, fps, frame_size)
        # cv2.CAP_PROP_FOURCC

    def start_record(self, filename, fps, frame_size):


# def record():

    def view_2d(self, numpy_image_2d):
        image = numpy_image_2d[:, :, :3]
        # Note the fourth parameter of QtGui.QImage below, which means how many bytes are in each line of the image. If it is not set, the image will sometimes be crooked, so it must be set
        image = QImage(image.data.tobytes(), image.shape[1], image.shape[0], QImage.Format_RGB888).rgbSwapped()
        pix = QPixmap(image).scaled(self.view_label.width(), self.view_label.height())
        self.view_label.setPixmap(pix)

        

def main():
    app = QApplication(sys.argv)
    example = Recorder()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()