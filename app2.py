import os
import pathlib

import gradio as gr
import torch
from PIL import Image

# 设置工作目录和下载必需文件
repo_dir = pathlib.Path("Thin-Plate-Spline-Motion-Model").absolute()
if not repo_dir.exists():
    os.system("git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model")
os.chdir(repo_dir.name)
if not (repo_dir / "checkpoints").exists():
    os.system("mkdir checkpoints")
if not (repo_dir / "checkpoints/vox.pth.tar").exists():
    os.system("gdown 1-CKOjv_y_TzNe-dwQsjjeVxJUuyBAb5X -O checkpoints/vox.pth.tar")


def inference(img, vid):
    if not os.path.exists('temp'):
        os.system('mkdir temp')

    img.save("temp/image.jpg", "JPEG")
    if torch.cuda.is_available():
        os.system(
            f"python demo.py --config config/vox-256.yaml --checkpoint ./checkpoints/vox.pth.tar --source_image temp/image.jpg --driving_video {vid} --result_video temp/result.mp4")
    else:
        os.system(
            f"python demo.py --config config/vox-256.yaml --checkpoint ./checkpoints/vox.pth.tar --source_image temp/image.jpg' --driving_video {vid} --result_video temp/result.mp4 --cpu")
    return './temp/result.mp4'


def main():
    with gr.Blocks() as demo:
        with gr.Tab("Video Generation"):
            with gr.Row():
                input_image = gr.Image(label="Input Image", type="pil")
                driving_video = gr.Video(label="Driving Video", format="mp4")
                generate_button = gr.Button("Generate")
                result_video = gr.Video(label="Result Video")

            generate_button.click(
                fn=inference,
                inputs=[input_image, driving_video],
                outputs=result_video
            )

        with gr.Tab("Audio Generation"):
            with gr.Row():
                gr.Markdown("### Audio generation feature is not yet implemented.")

    demo.launch()


if __name__ == '__main__':
    main()
