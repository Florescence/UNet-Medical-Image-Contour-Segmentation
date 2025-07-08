import os
import shutil
import logging
import argparse
import subprocess
from pathlib import Path


# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('seg_process.log'),
            logging.StreamHandler()
        ]
    )


# 创建中间目录
def create_work_dirs(root_dir):
    dirs = {
        "raw_png": os.path.join(root_dir, "1_raw_png"),  # 1.RAW转PNG
        "normalized_png": os.path.join(root_dir, "2_normalized_png"),  # 2.归一化512x512
        "pred_masks": os.path.join(root_dir, "3_pred_masks"),  # 3.预测掩码
        "denormalized_masks": os.path.join(root_dir, "4_denormalized_masks"),  # 4.反归一化掩码
        "json_results": os.path.join(root_dir, "5_json_results")  # 5.轮廓JSON和覆盖图
    }
    for dir_path in dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dirs


# 步骤1：RAW转PNG
def step_raw_to_png(input_raw, width, height, output_png_dir, **raw_kwargs):
    logging.info("===== 开始步骤1：RAW转PNG =====")
    # 调用raw2png.py的命令行模式（确保raw2png.py支持命令行传参）
    cmd = [
        "python", "utils/raw2png.py",
        input_raw,
        "--output", output_png_dir,
        "--width", str(width),
        "--height", str(height),
        "--percentile", str(raw_kwargs.get("percentile", 0.5))
    ]
    if raw_kwargs.get("flip_vertical"):
        cmd.append("--flip-vertical")
    if raw_kwargs.get("flip_horizontal"):
        cmd.append("--flip-horizontal")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logging.info(result.stdout)

    if not os.listdir(output_png_dir):
        raise RuntimeError("步骤1未生成任何PNG文件，终止流程")
    logging.info(f"步骤1完成：PNG保存至 {output_png_dir}")
    return output_png_dir


# 步骤2：PNG归一化到512x512
def step_normalize_png(input_png_dir, output_norm_dir, original_sizes_json, target_size=512):
    logging.info("===== 开始步骤2：PNG归一化到512x512 =====")
    # 调用png_normalize.py的命令行模式
    cmd = [
        "python", "utils/png_normalize.py",
        "--input-dir", input_png_dir,
        "--output-dir", output_norm_dir,
        "--target-size", str(target_size),
        "--original-sizes", original_sizes_json
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logging.info(result.stdout)

    if not os.listdir(output_norm_dir):
        raise RuntimeError("步骤2未生成任何归一化PNG，终止流程")
    logging.info(f"步骤2完成：归一化PNG保存至 {output_norm_dir}，原始尺寸记录至 {original_sizes_json}")
    return output_norm_dir, original_sizes_json


# 步骤3：轮廓预测
def step_predict_mask(input_norm_dir, output_pred_dir, model_path, classes, **predict_kwargs):
    logging.info("===== 开始步骤3：轮廓预测 =====")
    # 收集归一化后的PNG文件
    norm_pngs = [os.path.join(input_norm_dir, f) for f in os.listdir(input_norm_dir) if f.endswith(".png")]
    if not norm_pngs:
        raise RuntimeError("步骤3未找到归一化PNG，终止流程")

    # 调用predict.py的命令行模式
    cmd = [
        "python", "predict.py",
        "--model", model_path,
        "--input", *norm_pngs,
        "--output",
        *[os.path.join(output_pred_dir, f"{os.path.splitext(os.path.basename(f))[0]}_pred_mask.png") for f in
          norm_pngs],
        "--classes", str(classes),
        "--mask-threshold", str(predict_kwargs.get("mask_threshold", 0.5)),
        "--scale", str(predict_kwargs.get("scale", 0.5))
    ]
    if predict_kwargs.get("postprocess"):
        cmd.extend([
            "--postprocess",
            "--min-area", str(predict_kwargs.get("min_area", 4000)),
            "--morph-kernel", str(predict_kwargs.get("morph_kernel", 3))
        ])
    if predict_kwargs.get("bilinear"):
        cmd.append("--bilinear")

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logging.info(result.stdout)

    if not os.listdir(output_pred_dir):
        raise RuntimeError("步骤3未生成任何预测掩码，终止流程")
    logging.info(f"步骤3完成：预测掩码保存至 {output_pred_dir}")
    return output_pred_dir


# 步骤4：掩码反归一化
def step_denormalize_mask(input_pred_dir, output_denorm_dir, original_sizes_json):
    logging.info("===== 开始步骤4：掩码反归一化 =====")
    # 调用png_denormalize.py的命令行模式
    cmd = [
        "python", "utils/png_denormalize.py",
        "--input-dir", input_pred_dir,
        "--output-dir", output_denorm_dir,
        "--original-sizes", original_sizes_json
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logging.info(result.stdout)

    if not os.listdir(output_denorm_dir):
        raise RuntimeError("步骤4未生成任何反归一化掩码，终止流程")
    logging.info(f"步骤4完成：反归一化掩码保存至 {output_denorm_dir}")
    return output_denorm_dir


# 步骤5：Mask转Polygon（调用修改后的mask2polygon.py）
def step_mask_to_polygon(input_denorm_mask_dir, output_json_dir, width, height):
    logging.info("===== 开始步骤5：Mask转Polygon =====")
    # 调用mask2polygon.py的命令行模式（仅传递宽高参数）
    cmd = [
        "python", "utils/mask2polygon.py",
        "--mask-dir", input_denorm_mask_dir,
        "--output-dir", output_json_dir,
        "--width", str(width),
        "--height", str(height)
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logging.info(result.stdout)

    if not os.listdir(output_json_dir):
        raise RuntimeError("步骤5未生成任何JSON文件，终止流程")
    logging.info(f"步骤5完成：轮廓JSON和覆盖图保存至 {output_json_dir}")
    return output_json_dir


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="端到端RAW图像轮廓提取流程")
    # 基础参数
    parser.add_argument("input_raw", help="输入RAW文件路径或目录")
    parser.add_argument("--output-root", "-o", default="./seg_results", help="输出结果根目录")
    parser.add_argument("--width", "-w", type=int, required=True, help="RAW图像宽度")
    parser.add_argument("--height", "-h", type=int, required=True, help="RAW图像高度")
    parser.add_argument("--model", "-m", required=True, help="预测模型路径(.pth)")

    # 步骤1参数（raw2png）
    parser.add_argument("--raw-percentile", type=float, default=0.5, help="RAW转PNG的百分位数")
    parser.add_argument("--flip-vertical", action="store_true", help="RAW转PNG时垂直翻转")
    parser.add_argument("--flip-horizontal", action="store_true", help="RAW转PNG时水平翻转")

    # 步骤2参数（归一化）
    parser.add_argument("--norm-size", type=int, default=512, help="归一化目标尺寸")

    # 步骤3参数（预测）
    parser.add_argument("--classes", "-c", type=int, default=3, help="预测类别数")
    parser.add_argument("--postprocess", "-p", action="store_true", help="预测后处理")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="预测掩码阈值")
    parser.add_argument("--min-area", type=int, default=4000, help="后处理最小连通域面积")
    parser.add_argument("--morph-kernel", type=int, default=3, help="后处理形态学核大小")

    args = parser.parse_args()

    # 创建工作目录
    work_dirs = create_work_dirs(args.output_root)
    original_sizes_json = os.path.join(args.output_root, "original_sizes.json")  # 原始尺寸记录文件

    try:
        # 执行流程
        raw_png_dir = step_raw_to_png(
            input_raw=args.input_raw,
            width=args.width,
            height=args.height,
            output_png_dir=work_dirs["raw_png"],
            percentile=args.raw_percentile,
            flip_vertical=args.flip_vertical,
            flip_horizontal=args.flip_horizontal
        )

        norm_png_dir, sizes_json = step_normalize_png(
            input_png_dir=raw_png_dir,
            output_norm_dir=work_dirs["normalized_png"],
            original_sizes_json=original_sizes_json,
            target_size=args.norm_size
        )

        pred_mask_dir = step_predict_mask(
            input_norm_dir=norm_png_dir,
            output_pred_dir=work_dirs["pred_masks"],
            model_path=args.model,
            classes=args.classes,
            postprocess=args.postprocess,
            mask_threshold=args.mask_threshold,
            min_area=args.min_area,
            morph_kernel=args.morph_kernel
        )

        denorm_mask_dir = step_denormalize_mask(
            input_pred_dir=pred_mask_dir,
            output_denorm_dir=work_dirs["denormalized_masks"],
            original_sizes_json=sizes_json
        )

        json_result_dir = step_mask_to_polygon(
            input_denorm_mask_dir=denorm_mask_dir,
            output_json_dir=work_dirs["json_results"],
            width=args.width,  # 传递宽高参数到mask2polygon
            height=args.height
        )

        logging.info("===== 全流程完成 =====")
        logging.info(f"最终结果目录：{json_result_dir}")
        logging.info("流程成功结束")

    except Exception as e:
        logging.error(f"流程失败：{str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()