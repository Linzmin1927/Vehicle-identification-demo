import sys 
from yolo import YOLO
from PIL import Image, ImageDraw
import numpy as np

class Detector_YOLO():
    #模型相关参数
    defaults = {
    "model_path": 'Yolo_V3/model_data/yolo.h5',           #模型路径
    "anchors_path": 'Yolo_V3/model_data/tiny_yolo_anchors.txt',#锚点框模板信息
    "classes_path": 'Yolo_V3/model_data/voc_classes.txt',#类别信息
    "score" : 0.05,
    "iou" : 0.3,
    "model_image_size" : (416, 416),
    "gpu_num" : 0}
    def __init__(self):
        self.detector = YOLO(self.defaults)   
        self.out_boxes = []
        self.out_scores = []
        self.out_classes = []
    
    def detect_img(self,image,is_detecting=True):
        '''探测接口，输入image 输出叠加识别框的r_image，及识别框的位置'''
        if(False == is_detecting):
            if(self.out_boxes == []):
                return image
            else:
                new_image = self.box_img(image,self.out_boxes, self.out_scores, self.out_classes)
                return new_image

        # r_image = self.detector.detect_image(image)
        out_data = self.detector.detect_box(image)
        if(None == out_data):
            return image
        else:
            out_boxes, out_scores, out_classes = out_data
            new_image = self.box_img(image,out_boxes, out_scores, out_classes)
            self.out_boxes = out_boxes
            self.out_scores = out_scores
            self.out_classes = out_classes
            return new_image
        
    def box_img(self,image,out_boxes, out_scores, out_classes):
        '''给图像加框'''
        print("box_img",type(image))
        thickness = (image.size[0] + image.size[1]) // 300
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.detector.class_names[c]
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
                    outline= self.detector.colors[c])
            del draw 
        # print("box_img",type(image))
        return image

    
    def close(self):
        self.detector.close_session()


# detector = Detector_YOLO()
# image = Image.open("test.jpg")
# r_image = detector.detect_img(image)
# r_image[0].show()