import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':

    'change to your own path!!!'
    model = YOLO(model=r'E:\deeplearning\NavYolo\new-yolov8\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml')

    '加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升'
    #model.load('yolov8n.pt')
    model.train(data=r'voc2007.yaml',
                imgsz=640,
                epochs=100,
                batch=32,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/new',
                name='yolov8',
                single_cls=False,
                cache=True,
                )
