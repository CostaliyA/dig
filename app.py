import os
import pathlib

import gradio as gr
import torch
from PIL import Image

repo_dir = pathlib.Path("Thin-Plate-Spline-Motion-Model").absolute()
if not repo_dir.exists():
    os.system("git clone https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model")
os.chdir(repo_dir.name)
if not (repo_dir / "checkpoints").exists():
    os.system("mkdir checkpoints")
if not (repo_dir / "checkpoints/vox.pth.tar").exists():
    os.system("gdown 1-CKOjv_y_TzNe-dwQsjjeVxJUuyBAb5X -O checkpoints/vox.pth.tar")



title = "# Thin-Plate Spline Motion Model for Image Animation"
DESCRIPTION = '''### Gradio demo for <b>Thin-Plate Spline Motion Model for Image Animation</b>, CVPR 2022. <a href='https://arxiv.org/abs/2203.14367'>[Paper]</a><a href='https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model'>[Github Code]</a>
<img id="overview" alt="overview" src="https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model/raw/main/assets/vox.gif" />
'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=gradio-blocks.Image-Animation-using-Thin-Plate-Spline-Motion-Model" />'


def get_style_image_path(style_name: str) -> str:
    base_path = 'assets'
    filenames = {
        'source': 'source.png',
        'driving': 'driving.mp4',
    }
    return f'{base_path}/{filenames[style_name]}'


def get_style_image_markdown_text(style_name: str) -> str:
    url = get_style_image_path(style_name)
    return f'<img id="style-image" src="{url}" alt="style image">'


def update_style_image(style_name: str) -> dict:
    text = get_style_image_markdown_text(style_name)
    return gr.Markdown.update(value=text)


def inference(img, vid):
    if not os.path.exists('temp'):
        os.system('mkdir temp')

    img.save("temp/image.jpg", "JPEG")
    if torch.cuda.is_available():
        os.system(f"python demo.py --config config/vox-256.yaml --checkpoint ./checkpoints/vox.pth.tar --source_image temp/image.jpg --driving_video {vid} --result_video temp/result.mp4")
    else:
        os.system(f"python demo.py --config config/vox-256.yaml --checkpoint ./checkpoints/vox.pth.tar --source_image temp/image.jpg --driving_video {vid} --result_video temp/result.mp4 --cpu")
    return './temp/result.mp4'



def main():
    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(title)
        gr.Markdown(DESCRIPTION)

        with gr.Box():
            gr.Markdown('''## Step 1 (Provide Input Face Image)
- Drop an image containing a face to the **Input Image**.
    - If there are multiple faces in the image, use Edit button in the upper right corner and crop the input image beforehand.
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type="pil")

            with gr.Row():
                paths = sorted(pathlib.Path('assets').glob('*.png'))
                gr.Examples(inputs=[input_image],
                            examples=[[path.as_posix()] for path in paths])

        with gr.Box():
            gr.Markdown('''## Step 2 (Select Driving Video)
- Select **Style Driving Video for the face image animation**.
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        driving_video = gr.Video(label='Driving Video',
                                                 format="mp4")

            with gr.Row():
                paths = sorted(pathlib.Path('assets').glob('*.mp4'))
                gr.Examples(inputs=[driving_video],
                            examples=[[path.as_posix()] for path in paths])

        with gr.Box():
            gr.Markdown('''## Step 3 (Generate Animated Image based on the Video)
- Hit the **Generate** button. (Note: On cpu-basic, it takes ~ 10 minutes to generate final results.)
''')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        generate_button = gr.Button('Generate')

                with gr.Column():
                    result = gr.Video(label="Output")
        gr.Markdown(FOOTER)
        generate_button.click(fn=inference,
                              inputs=[
                                  input_image,
                                  driving_video
                              ],
                              outputs=result)

    demo.queue(max_size=10).launch()

if __name__ == '__main__':
    main()