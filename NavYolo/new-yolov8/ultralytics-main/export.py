# from ultralytics import YOLO
#
# # Load a model
# model = YOLO(r"E:\su\yolov8_improved\yolov8_improved\yolov8_pt\NavYolo.pt")  # load an official model
#
# # 导出成 ONNX
# model.export(format="onnx", opset=12, dynamic=True, simplify=True)

from ultralytics import YOLO

# 先构建模型（结构）
model = YOLO("yolov8n.yaml")  # 或者你的自定义 yolov8_improved.yaml

# 加载权重
model.load("yolov8n.pt")

# 导出 ONNX
model.export(format="onnx", opset=12, dynamic=True)