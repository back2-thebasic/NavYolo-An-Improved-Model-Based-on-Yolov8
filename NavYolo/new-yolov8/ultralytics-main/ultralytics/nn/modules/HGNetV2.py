import torch
from torch import nn
from timm.models import create_model
from torchinfo import summary


class HGNetv2(nn.Module):
    def __init__(self,number):
        super(HGNetv2, self).__init__()

        # 加载预训练的 RepViT 模型，提取特征图
        base_model = create_model('hgnetv2_b2.ssld_stage2_ft_in1k',
                     pretrained=True,
                     pretrained_cfg_overlay=dict(
                     file=r'E:/deeplearning/yolov8_improved/new-yolov8/ultralytics-main/bin/PP-HGNetV2/pytorch_model.bin'),
                     features_only=True)

        if number == 1:
            self.model = nn.Sequential(
                base_model.stem,
                base_model.stages_0,
                base_model.stages_1,


            )
        elif number == 2:
            self.model = nn.Sequential(
                base_model.stages_2
            )
        else:
            self.model = nn.Sequential(
                base_model.stages_3
            )

    def forward(self, x):
        return self.model(x)




if __name__ == '__main__':



    base_model = create_model('hgnetv2_b2.ssld_stage2_ft_in1k',
                              pretrained=True,
                              pretrained_cfg_overlay=dict(
                                  file=r'E:/deeplearning/yolov8_improved/new-yolov8/ultralytics-main/bin/PP-HGNetV2/pytorch_model.bin'),
                              features_only=True)

    model = nn.Sequential(
        base_model.stem,
        base_model.stages_0,
        base_model.stages_1,
        base_model.stages_2,
        base_model.stages_3,
    )

    # 打印模型的结构, 假设输入尺寸为 640x640
    summary(model, input_size=(1,3, 640, 640))  # 打印出模型的结构，输入通道是3，输入大小是640x640
    #
    # 测试代码
    x = torch.randn(1, 3, 640, 640)  # 模拟输入张量

    # 提取第一组（conv1, maxpool, stage2）
    model_group1 = HGNetv2(number=1)
    output_group1 = model_group1(x)
    print("Output of Group 1:", output_group1.shape)

    # 提取第二组（stage3）
    model_group2 = HGNetv2(number=2)
    output_group2 = model_group2(output_group1)  # 使用第一组的输出作为第二组的输入
    print("Output of Group 2:", output_group2.shape)

    # 提取第三组（stage4, conv5）
    model_group3 = HGNetv2(number=3)
    output_group3 = model_group3(output_group2)  # 使用第二组的输出作为第三组的输入
    print("Output of Group 3:", output_group3.shape)