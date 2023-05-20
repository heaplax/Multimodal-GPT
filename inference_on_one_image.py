import os

import gradio as gr
import torch
from PIL import Image

from mmgpt.models.builder import create_model_and_transforms
from app import Inferencer
from app import PromptGenerator

TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
response_split = "### Response:"

image_path = "/nobackup/users/zfchen/zt/clevr/teaser.jpg"
instruction = "How many objects are either small cylinders or red things?"

if __name__ == '__main__':
    llama_path = "checkpoints/llama-7b_hf"
    open_flamingo_path = "checkpoints/OpenFlamingo-9B/checkpoint.pt"
    finetune_path = "checkpoints/mmgpt-lora-v0-release.pt"
    inferencer = Inferencer(
        llama_path=llama_path,
        open_flamingo_path=open_flamingo_path,
        finetune_path=finetune_path)

    max_new_token = 512
    num_beams = 3
    temperature = 1
    top_k = 20
    top_p = 1
    do_sample = True

    image = image_path
    text = instruction

    prompt = TEMPLATE
    ai_prefix = "Response"
    user_prefix = "Instruction"
    seperator = "\n\n### "
    history_buffer = -1

    state = PromptGenerator()
    state.prompt_template = prompt
    state.ai_prefix = ai_prefix
    state.user_prefix = user_prefix
    state.sep = seperator
    state.buffer_size = history_buffer
    if image:
        state.add_message(user_prefix, (text, image))
    else:
        state.add_message(user_prefix, text)
    state.add_message(ai_prefix, None)
    inputs = state.get_prompt()
    image_paths = state.get_images()[-1:]

    inference_results = inferencer(inputs, image_paths, max_new_token,
                                   num_beams, temperature, top_k, top_p,
                                   do_sample)
    print("______________begin inference_results_____________")
    print(f"inference_results:{inference_results}")
    print(type(inference_results))
    print("______________end inference_results_____________")
    # ans = inference_results.strip("#").strip()
    # print(f"ans:{ans}")