import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import time
import copy
import random
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
from peft import PeftModel
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from transformers import get_linear_schedule_with_warmup
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def ChatCompletion(prompt, model, tokenizer):
    messages = [
    {
        "role": "user",
         "content": prompt
    }
    ]
    model.eval()
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(tokenized_chat, max_new_tokens=1024)
    pred = tokenizer.decode(outputs[0][tokenized_chat.shape[1]:])
    return pred


def main():
    path = f'./data/t2t_test_fixed.csv'   # change path here
    print(f'input file: {path}')
    test_df = pd.read_csv(path)
    inputs = test_df['t2t_input'].tolist()
    targets = test_df['t2t_target'].tolist()
    langs = test_df['language'].tolist()
    for i, lang in enumerate(langs):
        if lang == 'JS':
            langs[i] = 'JavaScript'
        elif lang == 'GO':
            langs[i] = 'Go'
    base_model = "../deepseek-coder-6.7b-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    T1 = time.perf_counter()
    pred_msg_list = []
    MAGICODER_PROMPT = """{instruction}
    """
    for idx in tqdm(range(len(inputs))):
        instruction = "You are an expert software developer in {}.\
                    You always want to improve your code to have higer quality.\
                    Your task is to repair the given vulnerable C/C++/C#/Java/JavaScript/Go/Python function.\
                    Please only generate the fixed code without your explanation. Here is a given vulnerable function: {}".format(langs[idx], inputs[idx])
        prompt = MAGICODER_PROMPT.format(instruction=instruction)
        pred_msg = ChatCompletion(prompt, model, tokenizer)
        pred_msg_list.append(pred_msg)
    T2 = time.perf_counter()
    print('Inference Time Total: %s s' % (T2 - T1))
    results_dict = {'outputs': pred_msg_list}
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.to_csv(f'./data/deepseek_zsp.csv', index=False)

if __name__ == "__main__":
    main()