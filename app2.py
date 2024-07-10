import os
import pathlib

import gradio as gr
import torch
import os, sys
import tempfile
from SadTalker.src.gradio_demo import SadTalker


def get_source_image(image):
    return image


try:
    import webui  # in webui

    in_webui = True
except:
    in_webui = False


def toggle_audio_file(choice):
    if choice == False:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def ref_video_fn(path_of_ref_video):
    if path_of_ref_video is not None:
        return gr.update(value=True)
    else:
        return gr.update(value=False)


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
            with gr.Group():
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

            with gr.Group():
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

            with gr.Group():
                gr.Markdown('''## Step 3 (Generate Animated Image based on the Video)
            - Hit the **Generate** button. (Note: On cpu-basic, it takes ~ 10 minutes to generate final results.)
            ''')
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            generate_button = gr.Button('Generate')

                    with gr.Column():
                        result = gr.Video(label="Output")

            generate_button.click(fn=inference,
                                  inputs=[
                                      input_image,
                                      driving_video
                                  ],
                                  outputs=result)

        with gr.Tab("Audio Generation"):
            with gr.Row():
                sad_talker = SadTalker(lazy_load=False)
                # tts_talker = TTSTalker()

                with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
                    with gr.Row():
                        with gr.Column(variant='panel'):
                            with gr.Tabs(elem_id="sadtalker_source_image"):
                                with gr.TabItem('Source image'):
                                    with gr.Row():
                                        source_image = gr.Image(label="Source image", type="filepath",
                                                                elem_id="img2img_image")

                            with gr.Tabs(elem_id="sadtalker_driven_audio"):
                                with gr.TabItem('Driving Methods'):
                                    gr.Markdown(
                                        "Possible driving combinations: <br> 1. Audio only 2. Audio/IDLE Mode + Ref Video(pose, blink, pose+blink) 3. IDLE Mode only 4. Ref Video only (all) ")

                                    with gr.Row():
                                        driven_audio = gr.Audio(label="Input audio", type="filepath")
                                        driven_audio_no = gr.Audio(label="Use IDLE mode, no audio is required",
                                                                   type="filepath", visible=False)

                                        with gr.Column():
                                            use_idle_mode = gr.Checkbox(label="Use Idle Animation")
                                            length_of_audio = gr.Number(value=5,
                                                                        label="The length(seconds) of the generated video.")
                                            use_idle_mode.change(toggle_audio_file, inputs=use_idle_mode,
                                                                 outputs=[driven_audio, driven_audio_no])  # todo

                                    with gr.Row():
                                        ref_video = gr.Video(label="Reference Video", elem_id="vidref")

                                        with gr.Column():
                                            use_ref_video = gr.Checkbox(label="Use Reference Video")
                                            ref_info = gr.Radio(['pose', 'blink', 'pose+blink', 'all'], value='pose',
                                                                label='Reference Video',
                                                                info="How to borrow from reference Video?((fully transfer, aka, video driving mode))")

                                        ref_video.change(ref_video_fn, inputs=ref_video,
                                                         outputs=[use_ref_video])  # todo

                        with gr.Column(variant='panel'):
                            with gr.Tabs(elem_id="sadtalker_checkbox"):
                                with gr.TabItem('Settings'):
                                    gr.Markdown(
                                        "need help? please visit our [[best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md)] for more detials")
                                    with gr.Column(variant='panel'):
                                        # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                                        # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                                        with gr.Row():
                                            pose_style = gr.Slider(minimum=0, maximum=45, step=1, label="Pose style",
                                                                   value=0)  #
                                            exp_weight = gr.Slider(minimum=0, maximum=3, step=0.1,
                                                                   label="expression scale", value=1)  #
                                            blink_every = gr.Checkbox(label="use eye blink", value=True)

                                        with gr.Row():
                                            size_of_image = gr.Radio([256, 512], value=256,
                                                                     label='face model resolution',
                                                                     info="use 256/512 model?")  #
                                            preprocess_type = gr.Radio(['crop', 'resize', 'full', 'extcrop', 'extfull'],
                                                                       value='crop', label='preprocess',
                                                                       info="How to handle input image?")

                                        with gr.Row():
                                            is_still_mode = gr.Checkbox(
                                                label="Still Mode (fewer head motion, works with preprocess `full`)")
                                            facerender = gr.Radio(['facevid2vid', 'pirender'], value='facevid2vid',
                                                                  label='facerender', info="which face render?")

                                        with gr.Row():
                                            batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10,
                                                                   value=1)
                                            enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")

                                        submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                            with gr.Tabs(elem_id="sadtalker_genearted"):
                                gen_video = gr.Video(label="Generated video", format="mp4")

                    submit.click(
                        fn=sad_talker.test,
                        inputs=[source_image,
                                driven_audio,
                                preprocess_type,
                                is_still_mode,
                                enhancer,
                                batch_size,
                                size_of_image,
                                pose_style,
                                facerender,
                                exp_weight,
                                use_ref_video,
                                ref_video,
                                ref_info,
                                use_idle_mode,
                                length_of_audio,
                                blink_every
                                ],
                        outputs=[gen_video],
                    )

                    with gr.Row():
                        examples = [
                            [
                                'examples/source_image/full_body_1.png',
                                'examples/driven_audio/bus_chinese.wav',
                                'crop',
                                True,
                                False
                            ],
                            [
                                'examples/source_image/full_body_2.png',
                                'examples/driven_audio/japanese.wav',
                                'crop',
                                False,
                                False
                            ],
                            [
                                'examples/source_image/full3.png',
                                'examples/driven_audio/deyu.wav',
                                'crop',
                                False,
                                True
                            ],
                            [
                                'examples/source_image/full4.jpeg',
                                'examples/driven_audio/eluosi.wav',
                                'full',
                                False,
                                True
                            ],
                            [
                                'examples/source_image/full4.jpeg',
                                'examples/driven_audio/imagine.wav',
                                'full',
                                True,
                                True
                            ],
                            [
                                'examples/source_image/full_body_1.png',
                                'examples/driven_audio/bus_chinese.wav',
                                'full',
                                True,
                                False
                            ],
                            [
                                'examples/source_image/art_13.png',
                                'examples/driven_audio/fayu.wav',
                                'resize',
                                True,
                                False
                            ],
                            [
                                'examples/source_image/art_5.png',
                                'examples/driven_audio/chinese_news.wav',
                                'resize',
                                False,
                                False
                            ],
                            [
                                'examples/source_image/art_5.png',
                                'examples/driven_audio/RD_Radio31_000.wav',
                                'resize',
                                True,
                                True
                            ],
                        ]
                        gr.Examples(examples=examples,
                                    inputs=[
                                        source_image,
                                        driven_audio,
                                        preprocess_type,
                                        is_still_mode,
                                        enhancer],
                                    outputs=[gen_video],
                                    fn=sad_talker.test,
                                    cache_examples=os.getenv('SYSTEM') == 'spaces')  #

    demo.queue(max_size=10).launch(share=True)


if __name__ == '__main__':
    main()
