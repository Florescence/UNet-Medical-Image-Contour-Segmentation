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
def step_raw_to_png(input_raw, output_png_dir, width, height, window_width, window_length):
    logging.info("===== 开始步骤1：RAW转PNG =====")
    # 调用raw2png.py的命令行模式（确保raw2png.py支持命令行传参）
    cmd = [
        "python", "utils/raw2png.py",
        "--input", str(input_raw),
        "--output", str(output_png_dir),
        "--width", str(width),
        "--height", str(height),
        "--window-width", str(window_width),
        "--window-length", str(window_length)
    ]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logging.info(result.stdout)

    if result.stderr:
        logging.error(f"RAW转PNG错误: {result.stderr}")  # 打印错误输出
    if result.returncode != 0:
        raise RuntimeError(f"raw2png.py执行失败，返回码: {result.returncode}")

    if not os.listdir(output_png_dir):
        raise RuntimeError("步骤1未生成任何PNG文件，终止流程")
    logging.info(f"步骤1完成：PNG保存至 {output_png_dir}")
    return output_png_dir


# 步骤2：PNG归一化到512x512
def step_normalize_png(input_png_dir, output_norm_dir):
    logging.info("===== 开始步骤2：PNG归一化到512x512 =====")
    # 调用png_normalize.py的命令行模式
    cmd = [
        "python", "utils/png_normalize.py",
        "--input", str(input_png_dir),
        "--output", str(output_norm_dir)
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logging.info(result.stdout)

    if not os.listdir(output_norm_dir):
        raise RuntimeError("步骤2未生成任何归一化PNG，终止流程")
    logging.info(f"步骤2完成：归一化PNG保存至 {output_norm_dir}")
    return output_norm_dir


# 步骤3：轮廓预测
def step_predict_mask(input_norm_dir, output_pred_dir, model_path):
    logging.info("===== 开始步骤3：轮廓预测 =====")
    # 收集归一化后的PNG文件
    norm_pngs = [os.path.join(input_norm_dir, f) for f in os.listdir(input_norm_dir) if f.endswith(".png")]
    if not norm_pngs:
        raise RuntimeError("步骤3未找到归一化PNG，终止流程")

    # 调用predict.py的命令行模式
    cmd = [
        "python", "predict.py",
        "--model", str(model_path),
        "--input", str(input_norm_dir),
        "--output", str(output_pred_dir),
    ]

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
        "--input", str(input_pred_dir),
        "--output", str(output_denorm_dir),
        "--json", str(original_sizes_json)
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    logging.info(result.stdout)

    if not os.listdir(output_denorm_dir):
        raise RuntimeError("步骤4未生成任何反归一化掩码，终止流程")
    logging.info(f"步骤4完成：反归一化掩码保存至 {output_denorm_dir}")
    return output_denorm_dir


# 步骤5：Mask转Polygon（调用修改后的mask2polygon.py）
def step_mask_to_polygon(input_denorm_mask_dir, output_json_dir, original_sizes_json):
    logging.info("===== 开始步骤5：Mask转Polygon =====")
    # 调用mask2polygon.py的命令行模式（仅传递宽高参数）
    cmd = [
        "python", "utils/mask2polygon.py",
        "--input", str(input_denorm_mask_dir),
        "--output", str(output_json_dir),
        "--json", str(original_sizes_json)
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
    parser.add_argument("--input-raw", help="输入RAW文件路径或目录")
    parser.add_argument("--output-root", "-o", default="seg_results", help="输出结果根目录")

    # 步骤1参数（raw2png）
    parser.add_argument("--width", type=int, required=True, help="RAW图像宽度")
    parser.add_argument("--height", type=int, required=True, help="RAW图像高度")
    parser.add_argument("--window-width", "-ww", type=int, required=True, help="RAW<UNK>")
    parser.add_argument("--window-length", "-wl", type=int, required=True, help="RAW<UNK>")
    parser.add_argument("--model", "-m", required=True, help="预测模型路径(.pth)")

    args = parser.parse_args()

    # 创建工作目录
    work_dirs = create_work_dirs(args.output_root)
    print(work_dirs["raw_png"])
    original_sizes_json = os.path.join(work_dirs["normalized_png"], "original_sizes.json")  # 原始尺寸记录文件

    try:
        # 执行流程
        raw_png_dir = step_raw_to_png(
            input_raw=args.input_raw,
            width=args.width,
            height=args.height,
            output_png_dir=work_dirs["raw_png"],
            window_width=args.window_width,
            window_length=args.window_length,
        )

        norm_png_dir = step_normalize_png(
            input_png_dir=raw_png_dir,
            output_norm_dir=work_dirs["normalized_png"]
        )

        pred_mask_dir = step_predict_mask(
            input_norm_dir=norm_png_dir,
            output_pred_dir=work_dirs["pred_masks"],
            model_path=args.model
        )

        denorm_mask_dir = step_denormalize_mask(
            input_pred_dir=pred_mask_dir,
            output_denorm_dir=work_dirs["denormalized_masks"],
            original_sizes_json=original_sizes_json
        )

        json_result_dir = step_mask_to_polygon(
            input_denorm_mask_dir=denorm_mask_dir,
            output_json_dir=work_dirs["json_results"],
            original_sizes_json=original_sizes_json
        )

        logging.info("===== 全流程完成 =====")
        logging.info(f"最终结果目录：{json_result_dir}")
        logging.info("流程成功结束")

    except Exception as e:
        logging.error(f"流程失败：{str(e)}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()