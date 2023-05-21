import json
import os
def get_clevr_question():
    # clevr_path = "F:/work/CLEVR/CLEVR_v1.0/"
    clevr_path = "/nobackup/users/zfchen/zt/clevr/CLEVR_v1.0"
    question_path = os.path.join(clevr_path, "questions", "CLEVR_val_questions.json")
    with open(question_path, "r") as f:
        question_list = json.load(f)["questions"]
    res_list = []
    for question in question_list:
        res_list.append({
            "split": question["split"],
            "image_path": os.path.join(clevr_path, "images", question["split"], question["image_filename"]),
            "question": question["question"],
            "answer": question["answer"],
        })
    return res_list[0:10000]

if __name__ == "__main__":
    print(get_clevr_question()[0:100])