from ultralytics import YOLO

model = YOLO(r'C:\Users\Me\Desktop\csv\model\yolov6.pt')  # 加载模型
model.info(detailed=True)  # 打印详细信息
model.profile(imgsz=[640,640])
model.fuse()

# E:\su\yolov8_improved\yolov8_improved\new-yolov8\ultralytics-main\runs\yolo\yolov3\weights\best.pt
