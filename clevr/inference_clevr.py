import os

import clevr.load_clevr as lclevr
import gradio as gr
import torch
from PIL import Image
import json

from mmgpt.models.builder import create_model_and_transforms
from app import Inferencer
from app import PromptGenerator

TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
response_split = "### Response:"

output_path = "/nobackup/users/zfchen/zt/Multimodal-GPT/output.json"
# image_path = "/nobackup/users/zfchen/zt/clevr/teaser.jpg"
# instruction = "How many objects are either small cylinders or red things?"


def inference_one(image, text, state):
    max_new_token = 512
    num_beams = 3
    temperature = 1
    top_k = 20
    top_p = 1
    do_sample = True
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
    inference_results = inference_results.split("\n")[0]
    state.all_history = []
    return inference_results
    # print("______________begin inference_results_____________")
    # print(f"inference_results:{inference_results}")
    # print(type(inference_results))
    # print("______________end inference_results_____________")

if __name__ == '__main__':
    llama_path = "checkpoints/llama-7b_hf"
    open_flamingo_path = "checkpoints/OpenFlamingo-9B/checkpoint.pt"
    finetune_path = "checkpoints/mmgpt-lora-v0-release.pt"
    inferencer = Inferencer(
        llama_path=llama_path,
        open_flamingo_path=open_flamingo_path,
        finetune_path=finetune_path)

    # image = image_path
    # text = instruction

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

    path_info = {
        "clevr_path": "/nobackup/users/zfchen/zt/clevr/CLEVR_v1.0",
        "result_file_path": "/nobackup/users/zfchen/zt/Multimodal-GPT/result_file.json",
        "ann_file_path": "/nobackup/users/zfchen/zt/Multimodal-GPT/ann_file.json",
        "ques_file_path": "/nobackup/users/zfchen/zt/Multimodal-GPT/ques_file.json",
        "output_path": "/nobackup/users/zfchen/zt/Multimodal-GPT/output.json",
    }

    question_list = lclevr.get_clevr_question(path_info)
    response_list = []

    for i, question in enumerate(question_list):
        response_list.append(inference_one(question["image_path"], question["question"], state))
        if i % 10 == 0:
            print(f"processed {i} questions")
    lclevr.generate_output(path_info, response_list)
    # ans = inference_results.strip("#").strip()
    # print(f"ans:{ans}")