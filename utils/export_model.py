import torch
from unet import UNet, UNet_S

# 加载训练好的模型
model = UNet_S(n_channels=1, n_classes=3, bilinear=False)
state_dict = torch.load("../checkpoints/model_S_clean_only_512x512.pth", map_location=torch.device('cpu'))  # 强制加载到CPU
if 'mask_values' in state_dict:
    del state_dict['mask_values']
model.load_state_dict(state_dict)
model.eval()

# 【关键验证】确保模型在CPU上
if next(model.parameters()).is_cuda:
    print("警告：模型在GPU上，即将转移到CPU！")
    model = model.cpu()

# 导出为TorchScript（使用CPU示例输入）
example_input = torch.randn(1, 1, 512, 512)  # 确保输入在CPU上（未调用.cuda()）
traced_model = torch.jit.trace(model, example_input)

# 【关键验证】测试导出的模型是否能在CPU上推理
with torch.no_grad():
    output = traced_model(example_input)
    print("导出模型推理成功！输出形状：", output.shape)  # 应输出(1, 3, 512, 512)

traced_model.save("../unet_model.pt")