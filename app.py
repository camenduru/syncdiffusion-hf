"""
app.py
An interactive demo for text-guided panorama generation.
"""
import torch
import gradio as gr

from syncdiffusion.syncdiffusion_model import SyncDiffusion
from syncdiffusion.utils import seed_everything

def run_inference(
        prompt: str,  
        width: int = 2048,
        sync_weight: float = 20.0,
        sync_thres: int = 5,
        seed: int = 0
    ):
    # set device
    device = torch.device = torch.device("cuda")

    # set random seed
    seed_everything(seed)

    # load SyncDiffusion model
    syncdiffusion = SyncDiffusion(device, sd_version="2.0")

    img = syncdiffusion.sample_syncdiffusion(
        prompts = prompt,
        negative_prompts = "",
        height = height,
        width = width,
        num_inference_steps = 50,
        guidance_scale = 7.5,
        sync_weight = sync_weight,
        sync_decay_rate = sync_decay_rate,
        sync_freq = sync_freq,
        sync_thres = sync_thres,
        stride = 16
        )
    return [img]

if __name__=="__main__":
    title = "SyncDiffusion: Text-Guided Panorama Generation"

    description_text = '''
    This demo features text-guided panorama generation from our work <a href="https://arxiv.org/abs/2306.05178">SyncDiffusion: Coherent Montage via Synchronized Joint Diffusions, NeurIPS 2023</a>.  
    Please refer to our <a href="https://syncdiffusion.github.io/">project page</a> for details.
    '''

    # create UI        
    with gr.Blocks(title=title) as demo:

        # description of demo
        gr.Markdown(description_text)

        # inputs
        with gr.Row():
            with gr.Column():
                run_button = gr.Button(label="Generate")

                prompt = gr.Textbox(label="Text Prompt", value='a cinematic view of a castle in the sunset')
                width = gr.Slider(label="Width", minimum=512, maximum=4096, value=2048, step=128)
                sync_weight = gr.Slider(label="Sync Weight", minimum=0.0, maximum=30.0, value=20.0, step=5.0)
                sync_thres = gr.Slider(label="Sync Threshold (If N, apply SyncDiffusion for the first N steps)", minimum=0, maximum=50, value=5, step=1)
                seed = gr.Number(label="Seed", value=0)

            with gr.Column():
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')

        # display examples
        examples = gr.Examples(
            examples=[
                'a cinematic view of a castle in the sunset',
                'natural landscape in anime style illustration',
                'a photo of a lake under the northern lights',
            ],
            inputs=[prompt],
        )

        ips = [prompt, width, sync_weight, sync_thres, seed]
        run_button.click(fn=run_inference, inputs=ips, outputs=[result_gallery])

    demo.queue(max_size=30)
    demo.launch()