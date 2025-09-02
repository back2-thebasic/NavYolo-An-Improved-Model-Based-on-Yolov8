import torch
import torchvision.models as models
from torch import nn
from torchinfo import summary


class ShuffleNetV2(nn.Module):
    def __init__(self, number):
        super(ShuffleNetV2, self).__init__()
        self.model = None
        base_model = models.shufflenet_v2_x1_0(pretrained=True)  # 加载预训练的 ShuffleNetV2 模型

        if number == 1:
            self.model = nn.Sequential(
                base_model.conv1,
            )

        elif number == 2:

            self.model = nn.Sequential(
                base_model.maxpool,
            )

        elif number == 3:
            self.model = nn.Sequential(
                base_model.stage2
            )
        elif number == 4:
            self.model = nn.Sequential(
                base_model.stage3
            )
        else:
            self.model = nn.Sequential(
                base_model.stage4,
                base_model.conv5
            )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # 加载预训练的 ShuffleNetV2 模型
    model = models.shufflenet_v2_x1_0(pretrained=True)


    x = torch.randn(1, 3, 640, 640)

    summary(model, input_size=x.shape)


   # 测试代码
    x = torch.randn(1, 3, 640, 640)  # 模拟输入张量

    # 提取第一组
    model_group1 = ShuffleNetV2(number=1)
    output_group1 = model_group1(x)
    print("Output of Group 1:", output_group1.shape)

    # 提取第二组
    model_group2 = ShuffleNetV2(number=2)
    output_group2 = model_group2(output_group1)
    print("Output of Group 2:", output_group2.shape)

    # 提取第三组
    model_group3 = ShuffleNetV2(number=3)
    output_group3 = model_group3(output_group2)
    print("Output of Group 3:", output_group3.shape)

    # 提取第四组
    model_group4 = ShuffleNetV2(number=4)
    output_group4 = model_group4(output_group3)
    print("Output of Group 3:", output_group4.shape)

    # 提取第五组
    model_group5 = ShuffleNetV2(number=5)
    output_group5 = model_group5(output_group4)
    print("Output of Group 3:", output_group5.shape)