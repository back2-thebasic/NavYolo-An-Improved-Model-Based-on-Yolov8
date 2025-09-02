import torch
from ultralytics.nn.modules import ShuffleNetV2, VoVGSCSP, DetectDSConv

x = torch.randn(1, 3, 640, 640)

# 1. 测试 ShuffleNetV2 输出
print('Testing ShuffleNetV2...')
model = ShuffleNetV2(24, 1)  # 修正参数为你 YAML 里用的格式
try:
    y = model(x)
    print(f"ShuffleNetV2 output: {y.shape if isinstance(y, torch.Tensor) else [t.shape for t in y]}")
except Exception as e:
    print(f"ShuffleNetV2 failed: {e}")

# 2. 测试 VoVGSCSP 模块
print('\nTesting VoVGSCSP...')
x2 = torch.randn(1, 256, 80, 80)
neck = VoVGSCSP(256)
try:
    y2 = neck(x2)
    print(f"VoVGSCSP output: {y2.shape}")
except Exception as e:
    print(f"VoVGSCSP failed: {e}")

# 3. 测试 DetectDSConv
print('\nTesting DetectDSConv...')
nc = 80
ch = [256, 512, 1024]
detect = DetectDSConv(nc, ch)
x3 = [torch.randn(1, c, s, s) for c, s in zip(ch, [80, 40, 20])]
try:
    y3 = detect(x3)
    print(f"DetectDSConv output: {[t.shape for t in y3]}")
except Exception as e:
    print(f"DetectDSConv failed: {e}")
