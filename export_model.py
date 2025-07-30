import torch

from unet import UNet_S, UNet
from unet.unet_model import UNet_SA
from yolo.yolov8_seg_model import YOLOv8_Seg_S

# 加载训练好的模型
model = UNet_S(n_channels=1, n_classes=3, bilinear=False)
# model = YOLOv8_Seg_S(n_channels=1, n_classes=1)
state_dict = torch.load("checkpoints/model_S_boundary_512x512.pth")
if 'mask_values' in state_dict:
    del state_dict['mask_values']
model.load_state_dict(state_dict)
model.eval()

# 自动检测可用设备
device = torch.device('cuda' if torch.cuda.is_available() else '')
model = model.to(device)
print(f"模型已加载到 {device}")

# 🌟 修改1：使用任意尺寸的example_input（例如224x224）
# 实际导出时尺寸不重要，只要符合模型预期的输入格式即可
example_input = torch.randn(1, 1, 512, 512, dtype=torch.float32).to(device)  # 不再需要元组包装

# 定义输入输出名称
input_names = ["input"]
output_names = ["output"]

# 🌟 修改2：添加对height和width的动态轴支持
dynamic_axes = {
    "input": {0: "batch_size", 2: "height", 3: "width"},  # 第2和第3维（H/W）设为动态
    "output": {0: "batch_size", 2: "height", 3: "width"}  # 输出尺寸需与输入保持一致
}

# 导出模型到ONNX
torch.onnx.export(
    model,                      # 模型实例
    example_input,              # 示例输入
    "unet_model_bs.onnx",  # 🌟 修改3：使用新文件名，避免覆盖原模型
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes
)

# 验证模型推理
with torch.no_grad():
    output = model(example_input)
    print(f"模型在 {device} 上推理成功！输出形状：{output.shape}, {output.dtype}")

print("支持动态尺寸的模型已成功导出为ONNX格式！")