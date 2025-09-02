# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2

# Load a model
model = YOLO(model=r'E:\su\test_model\yolov8_2.pt',task='detect')  # load a pretrained model (recommended for training)


# # Train the model
# results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)

model.predict(source=r'E:\su\yolov8_improved\yolov8_improved\new-yolov8\ultralytics-main\ultralytics\datasets\voc2007\train\images\000804.jpg',save=True,show=True,save_dir=r'E:\su\predict_results')



