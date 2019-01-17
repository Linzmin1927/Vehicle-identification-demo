import sys 
from yolo import YOLO
from PIL import Image, ImageDraw
import numpy as np
import threading
import copy

# 建一个全局变量，方便多线程调用
    #模型相关参数
defaults = {
"model_path": 'Yolo_V3/model_data/yolo.h5',           #模型路径
"anchors_path": 'Yolo_V3/model_data/tiny_yolo_anchors.txt',#锚点框模板信息
"classes_path": 'Yolo_V3/model_data/voc_classes.txt',#类别信息
"score" : 0.05,
"iou" : 0.3,
"model_image_size" : (416, 416),
"gpu_num" : 0}
Yolo_detector = YOLO(defaults)   
is_detecting_run = False
out_boxes, out_scores, out_classes = None, None, None
R=threading.Lock()


class  DetectingThread (threading.Thread):
    def __init__(self, threadID, image):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.image = image
    def run(self):
        global Yolo_detector,is_detecting_run,out_boxes, out_scores, out_classes 
       # 获得锁，成功获得锁定后返回True
       # 可选的timeout参数不填时将一直阻塞直到获得锁定
       # 否则超时后将返回False
        R.acquire()
        is_detecting_run = True
        # 释放锁
        R.release()
        # 执行探测
        # print(type(self.image),self.image)
        out_data = Yolo_detector.detect_box(self.image)
        R.acquire()
        if out_data is None:
            out_boxes, out_scores, out_classes = None, None, None
        else:
            out_boxes, out_scores, out_classes = out_data
        is_detecting_run = False
        R.release()

class Detector_YOLO():

    def __init__(self):
        self.out_boxes = []
        self.out_scores = []
        self.out_classes = []
        self.class_names = Yolo_detector.class_names
        self.colors = Yolo_detector.colors
    
    def detect_img(self,image):
        '''探测接口，输入image 输出叠加识别框的r_image，及识别框的位置'''
        global Yolo_detector,g_image,is_detecting_run,out_boxes, out_scores, out_classes 

        if( False == is_detecting_run):#探测线程没有开始
            # print("start detecting!!!!")
            # g_image = copy.deepcopy(image)
            thread1 = DetectingThread(1, image)
            thread1.start()

        if out_boxes is None:
            return image
        else:
            new_image = self.box_img(image,out_boxes, out_scores, out_classes)
            return new_image

    def box_img(self,image,out_boxes, out_scores, out_classes):
        '''给图像加框'''
        # print("box_img",type(image))
        thickness = (image.size[0] + image.size[1]) // 300
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            # print(out_boxes)
            # print(box)
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)

            top, left, bottom, right = box
 
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline= self.colors[c])
            del draw 
        # print("box_img",type(image))
        return image

    
    def close(self):
        Yolo_detector.close_session()


# detector = Detector_YOLO()
# image = Image.open("test.jpg")
# r_image = detector.detect_img(image)
# r_image[0].show()