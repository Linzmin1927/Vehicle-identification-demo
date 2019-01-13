import xml.etree.ElementTree as ET
import os
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["plate"]

def convert_annotation( image_id, list_file):
    in_file = open('.\\data\\xml\\%s.xml'%(image_id), encoding='UTF-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

# 将图片名写入文档
file_dir = getcwd()
imglist = os.listdir(".\\data\\pic") 
list_file = open('.\\data\\datalist.txt', 'w')
for image in imglist:
    list_file.write(image+"\n")
list_file.close()

#将xml文件转为yolo训练文件
list_file = open('.\\data\\Yolo_train.txt', 'w')
for imagename in imglist:
    list_file.write('.\\data\\pic\\%s'%(imagename))
    convert_annotation(imagename[:-4], list_file)
    list_file.write('\n')
list_file.close()
