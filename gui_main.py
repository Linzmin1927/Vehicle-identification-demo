import time
import sys
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from cv2 import *
from detection_module import Detector_YOLO

class main_GUI(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        # GUI主界面

        pe = QPalette()
        pe.setColor(QPalette.Window,Qt.black)#设置label颜色
   

        self.setFixedSize(1200,800)

        self.pictureLabel = QLabel()
        self.pictureLabel.setFixedWidth(580)
        self.pictureLabel.setFixedHeight(600)
        self.pictureLabel.setAutoFillBackground(True)#设置背景充满，为设置背景颜色的必要条件
        # self.pictureLabel.setPalette(pe)
        # init_image = QPixmap("data\豫M00000.jpg").scaled(self.width()*2/3, self.height()*2/3)
        self.pictureLabel.setPalette(pe)

        self.openButton = QPushButton()
        self.openButton.setEnabled(True)
        self.openButton.setText("Open")
        self.openButton.clicked.connect(self.open_video)

        self.resultLabel = QLabel()
        self.resultLabel.setFixedWidth(580)
        self.resultLabel.setFixedHeight(200)
        init_image = QPixmap("data\豫M00000.jpg").scaled(self.width()/2, self.height()/2)
        self.resultLabel.setPixmap(init_image)
        # self.resultLabel.setPalette(pe)

        self.resultList = QListWidget()

        layout_H_1 = QHBoxLayout()
        layout_H_1.addStretch(1)
        layout_H_1.addWidget(self.openButton)
        layout_H_1.addStretch(1)

        layout_V = QVBoxLayout()
        layout_V.addWidget(self.pictureLabel)
        layout_V.addLayout(layout_H_1)

        layout_V_1 = QVBoxLayout()
        layout_V_1.addWidget(self.resultLabel)
        layout_V_1.addWidget(self.resultList)

        layout_final = QHBoxLayout()
        layout_final.addLayout(layout_V)
        layout_final.addLayout(layout_V_1)

        self.setLayout(layout_final)

        # timer 设置
        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(self.show_images)

        # video 初始设置，视频播放
        # self.playCapture = VideoCapture()
        self.playCapture = []

        # 测评探测网络初始化
        self.detector = Detector_YOLO()


    def open_video(self):
        video_url, filetype = QFileDialog.getOpenFileName(self,
                    "选取文件",
                    "./",
                    "All Files (*);;MP4 Files (*.mp4)") 
        print(video_url)
        if video_url == "":
            return
        #先读一帧看看效果
        self.playCapture = cv2.VideoCapture(video_url)
        if not  self.playCapture.isOpened():
            print("Couldn't open webcam or video")
            return
        video_FourCC    = int( self.playCapture.get(cv2.CAP_PROP_FOURCC))
        video_fps       =  self.playCapture.get(cv2.CAP_PROP_FPS)
        video_size      = (int( self.playCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int( self.playCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return_value, frame =  self.playCapture.read()
        image = Image.fromarray(frame)
        image_new = self.detector.detect_img(image)
        print(image_new)
        if(None==image_new):
            height, width = frame.shape[:2]
            print(type(frame),frame.shape[:2])
            if frame.ndim == 3:
                rgb = cvtColor(frame, COLOR_BGR2RGB)
            elif frame.ndim == 2:
                rgb = cvtColor(frame, COLOR_GRAY2BGR)
            temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
            temp_pixmap = QPixmap.fromImage(temp_image).scaled(self.pictureLabel.width(), self.pictureLabel.height())
            self.pictureLabel.setPixmap(temp_pixmap)  
        else:            
            temp_image = ImageQt(image_new)
            temp_pixmap = QPixmap.fromImage(temp_image).scaled(self.pictureLabel.width(), self.pictureLabel.height())
            self.pictureLabel.setPixmap(temp_pixmap)


        fps = self.playCapture.get(CAP_PROP_FPS)
        self.fps = fps
        self.count = 0
        self.frame_rate = 8  #多少帧识别一次
        print("fps:"+str(fps))
        self.timer.set_fps(fps)
        # self.timer.set_fps(3)
        # self.playCapture.release() #
        self.timer.start()

            # self.videoWriter = VideoWriter('*.mp4', VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, size)
    def show_images(self):
        self.count += 1 
        is_detecting = False
        if( 0 == self.count%self.frame_rate):
            is_detecting = True
        if self.playCapture.isOpened():
            success, frame = self.playCapture.read()
            if success:
                image = Image.fromarray(frame)
                new_image = self.detector.detect_img(image)
                # print(type(new_image))
                if new_image is None:
                # if(True):
                    height, width = frame.shape[:2]
                    if frame.ndim == 3:
                        rgb = cvtColor(frame, COLOR_BGR2RGB)
                    elif frame.ndim == 2:
                        rgb = cvtColor(frame, COLOR_GRAY2BGR)
                    temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
                    temp_pixmap = QPixmap.fromImage(temp_image).scaled(self.pictureLabel.width(), self.pictureLabel.height())
                    self.pictureLabel.setPixmap(temp_pixmap)   
                else:
                    temp_image = ImageQt(new_image)
                    temp_pixmap = QPixmap.fromImage(temp_image).scaled(self.pictureLabel.width(), self.pictureLabel.height())
                    self.pictureLabel.setPixmap(temp_pixmap)

            else:
                print("read failed, no frame data")
                self.timer.stop()
        else:
            print("open file or capturing device error, init again")
            self.reset()
        return
class Communicate(QObject):

    signal = pyqtSignal(str)


class VideoTimer(QThread):

    def __init__(self, frequent=20):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def set_fps(self, fps):
        self.frequent = fps


if __name__ == "__main__":
    mapp = QApplication(sys.argv)
    mw = main_GUI()
    mw.show()
    sys.exit(mapp.exec_())
