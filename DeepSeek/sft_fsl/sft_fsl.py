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
print(torch.cuda.is_available())
print(torch.cuda.device_count())
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


def main(idx):
    path = f'./data/deepseek_fs_prompt3.csv'
    test_df = pd.read_csv(path)
    prompt_list = test_df['sys_prompt']

    base_model = "../deepseek-coder-6.7b-instruct"   # Change model here
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    checkpoint_dir = "../checkpoints/deepseek"  # Chang save checkpoint path here
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    # model.to(torch.device('cuda:0'))
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # inference
    T1 = time.perf_counter()
    pred_msg_list = []
    # test case
    # prompt_list = prompt_list[:3]
    
    for prompt in tqdm(prompt_list):
        pred_msg = ChatCompletion(prompt, model, tokenizer)
        pred_msg_list.append(pred_msg)
    T2 = time.perf_counter()
    print('Inference Time Total: %s s' % (T2 - T1))

    results_dict = {'outputs': pred_msg_list}
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.to_csv(f'./data/deepseek_sft_fsp.csv', index=False)

if __name__ == "__main__":
    num = 0
    main(num)
