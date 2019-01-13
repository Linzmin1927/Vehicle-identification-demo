import sys 
from cv2 import *
from yolo import YOLO
from PIL import Image

class Detector_YOLO():
    #模型相关参数
    defaults = {
    "model_path": 'Yolo_V3/model_data/yolo.h5',           #模型路径
    "anchors_path": 'Yolo_V3/model_data/tiny_yolo_anchors.txt',#锚点框模板信息
    "classes_path": 'Yolo_V3/model_data/coco_classes.txt',#类别信息
    "score" : 0.3,
    "iou" : 0.45,
    "model_image_size" : (416, 416),
    "gpu_num" : 0}
    def __init__(self):
        self.detector = YOLO(self.defaults)   
    
    def detect_img(self,image):
        '''探测接口，输入image 输出叠加识别框的r_image，及识别框的位置'''
        object_pos=[]
        r_image = self.detector.detect_image(image)
        return r_image
    
    def close(self):
        self.detector.close_session()


# detector = Detector_YOLO()
# image = Image.open("test.jpg")
# r_image = detector.detect_img(image)
# r_image[0].show()