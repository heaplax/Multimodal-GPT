import clevr.load_clevr as lclevr

if __name__ == "__main__":
    path_info = {
        "clevr_path": "/nobackup/users/zfchen/zt/clevr/CLEVR_v1.0",
        "result_file_path": "F:/work//Multimodal-GPT/result_file.json",
        "ann_file_path": "F:/work//Multimodal-GPT/ann_file.json",
        "ques_file_path": "F:/work//Multimodal-GPT/ques_file.json",
        "output_path": "F:/work//Multimodal-GPT/output.json",
    }
    lclevr.eval_output(path_info)