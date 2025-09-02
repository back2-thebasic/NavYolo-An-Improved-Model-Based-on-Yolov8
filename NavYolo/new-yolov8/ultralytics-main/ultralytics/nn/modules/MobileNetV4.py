import timm
import torch
from torch import nn
from torchinfo import summary




class MobileNetV4(nn.Module):
    def __init__(self,number):
        super(MobileNetV4, self).__init__()

        base_model = timm.create_model('mobilenetv4_hybrid_medium', pretrained=True,
                                       pretrained_cfg_overlay=dict(
                                           file=r'E:\su\yolov8_improved\yolov8_improved\new-yolov8\ultralytics-main\bin\MobileNetV4\pytorch_model.bin'),
                                       features_only=True)


        if number == 1:
            self.model = nn.Sequential(
                base_model.conv_stem,
                base_model.bn1,
                base_model.act1,
            )
        elif number == 2:
            self.model = nn.Sequential(
                base_model.blocks[:1]

            )
        elif number == 3:
            self.model = nn.Sequential(
                base_model.blocks[1:2]

            )
        elif number == 4:
            self.model = nn.Sequential(
                base_model.blocks[2:3]
            )
        else:
            self.model = nn.Sequential(
                base_model.blocks[3:5]
            )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = timm.create_model('mobilenetv4_hybrid_medium', pretrained=True,
                                       pretrained_cfg_overlay=dict(
                                           file=r'E:\su\yolov8_improved\yolov8_improved\new-yolov8\ultralytics-main\bin\MobileNetV4\pytorch_model.bin'),
                                       features_only=True)

    #print(model)
    # 打印模型输出特征图大小
    summary(model, input_size=(1, 3, 640, 640))


    # 测试代码
    x = torch.randn(1, 3, 640, 640)  # 模拟输入张量

    # 提取第一组
    model_group1 = MobileNetV4(number=1)
    output_group1 = model_group1(x)
    print("Output of Group 1:", output_group1.shape)

    # 提取第二组
    model_group2 = MobileNetV4(number=2)
    output_group2 = model_group2(output_group1)
    print("Output of Group 2:", output_group2.shape)

    # 提取第三组
    model_group3 = MobileNetV4(number=3)
    output_group3 = model_group3(output_group2)
    print("Output of Group 3:", output_group3.shape)

    # 提取第四组
    model_group4 = MobileNetV4(number=4)
    output_group4 = model_group4(output_group3)
    print("Output of Group 3:", output_group4.shape)

    # 提取第五组
    model_group5 = MobileNetV4(number=5)
    output_group5 = model_group5(output_group4)
    print("Output of Group 3:", output_group5.shape)