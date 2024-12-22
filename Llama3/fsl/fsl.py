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

def ChatCompletion(instruction, content, model, tokenizer):
    messages = [
    {
        "role": "system",
        "content": instruction,
    },
    {
        "role": "user",
         "content": content
    }
    ]
    model.eval()
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(tokenized_chat, max_new_tokens=1024, pad_token_id = tokenizer.eos_token_id)
    pred = tokenizer.decode(outputs[0][tokenized_chat.shape[1]:])
    return pred


def main():
    path = f'./data/llama_fs_prompt3.csv'
    test_df = pd.read_csv(path)
    sys_prompt_list = test_df['sys_prompt']
    user_prompt_list = test_df['user_prompt']
    input_pairs = list(zip(sys_prompt_list, user_prompt_list))

    base_model = "../Meta-Llama-3-8B-Instruct"   # Change model here
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # inference
    T1 = time.perf_counter()
    pred_msg_list = []
    for sys_p, user_p in tqdm(input_pairs):
        pred_msg = ChatCompletion(sys_p, user_p, model, tokenizer)
        pred_msg_list.append(pred_msg)
    T2 = time.perf_counter()
    print('Inference Time Total: %s s' % (T2 - T1))

    results_dict = {'outputs': pred_msg_list}
    results_df = pd.DataFrame.from_dict(results_dict)
    results_df.to_csv(f'./data/llama3_8b_fsp.csv', index=False)

if __name__ == "__main__":
    main()
