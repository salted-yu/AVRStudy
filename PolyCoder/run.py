from __future__ import absolute_import, division, print_function
import argparse
import ast
import copy
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

cpu_cont = 16
logger = logging.getLogger(__name__)
IGNORE_INDEX = -100


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        self.tokenizer = tokenizer
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        source = df["source"].tolist()
        repair_target = df["target"].tolist()
        # 记录每条数据的语言
        # 判断是不是REEF数据集，通过最后的文件名是否包含reef来判断
        self.file_type = file_type
        if "reef" in file_path:
            languages = df['language'].tolist()
        else:
            # 和source一样长的C语言列表
            languages = ["C"] * len(source)
        for i in tqdm(range(len(source))):
            # for i in tqdm(range(3)):
            self.examples.append(
                self.convert_examples_to_features(source[i], repair_target[i], tokenizer, args, languages[i]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # return self.examples[i]
        if self.file_type == 'test':
            return torch.tensor(self.examples[i]["input_ids"]), self.examples[i]['label'], self.examples[i]['language']
        else:
            return torch.tensor(self.examples[i]["input_ids"]), torch.tensor(
                self.examples[i]['attention_mask']), torch.tensor(self.examples[i]['label'])

    def convert_examples_to_features(self, source, repair_target, tokenizer, args, language):
        if self.file_type == 'test':
            inputs = tokenizer.encode(source, max_length=args.decoder_block_size - args.output_size - 1,
                                      truncation=True)
            start_token = tokenizer.encode('<S2SV_ModStart>')
            inputs = inputs + start_token
            return {
                "input_ids": inputs,
                "label": repair_target,
                'language': language
            }
        else:
            # 计算两个输入的最大长度
            source_token = tokenizer(source).input_ids
            tgt_token = tokenizer(repair_target).input_ids
            tgt_len = max(args.output_size, args.decoder_block_size - len(source_token))
            tgt_len = min(tgt_len, len(tgt_token))
            tgt_token = tgt_token[:tgt_len]
            # 在输入的最后加上eos_token
            src_len = min(len(source_token), args.decoder_block_size - len(tgt_token) - 1)
            src_token = source_token[:src_len]

            # 生成数据，至少留一个token 给 eos_token
            input_ids = src_token + tgt_token + [tokenizer.eos_token_id] * (
                        args.decoder_block_size - len(src_token) - len(tgt_token))
            label = [IGNORE_INDEX] * len(src_token) + tgt_token + [tokenizer.eos_token_id] * (
                        args.decoder_block_size - len(src_token) - len(tgt_token))
            attention_mask = [1 if i != tokenizer.pad_token_id else 0 for i in input_ids]

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": label,  # 使用labels键存储输出的token ids
                'language': language
            }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    # 计算总的step和warmup_steps
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader) // args.gradient_accumulation_steps
    args.warmup_steps = int(args.max_steps * 0.1)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    model.to(args.device)
    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 114514

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_loss = 0
        train_loss = 0
        tr_num = 0
        for step, batch in enumerate(bar):
            input_ids, attention_mask, label = [x.to(args.device) for x in batch]
            model.train()

            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            loss.backward()

            tr_loss += loss.item()
            train_loss += loss.item()
            tr_num += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                avg_loss = round(train_loss / tr_num, 5)
                bar.set_description(f"epoch {idx} loss {avg_loss}")

                if global_step % args.save_steps == 0:
                    # Placeholder for evaluation
                    eval_loss = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  " + "*" * 20)
                        logger.info(f"  Best Loss: {round(best_loss, 4)}")
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, args.model_name)
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(f"Saving model checkpoint to {output_dir}")


def clean_tokens(tokens, filter_tokens=None):
    if filter_tokens is not None:
        for t in filter_tokens:
            if t is None:
                continue
            tokens = tokens.replace(t, "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, attention_mask, label) = [x.to(args.device) for x in batch]
        loss = model(input_ids, attention_mask=attention_mask, labels=label).loss
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss / num, 5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss


def test(args, model, tokenizer, test_dataset):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    accuracy = []
    raw_predictions = []
    correct_prediction = ""
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    record_data = []
    for batch in bar:
        correct_pred = False
        (input_ids, targets, languages) = [x.to(args.device) if isinstance(x, torch.Tensor) else x for x in batch]
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(args.device)
        with torch.no_grad():
            beam_outputs = model.generate(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          do_sample=False,  # disable sampling to test if batching affects output
                                          num_beams=args.num_beams,
                                          max_new_tokens=args.output_size,
                                          num_return_sequences=args.num_beams,
                                          pad_token_id=tokenizer.eos_token_id
                                          )
        source_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        # 去掉最后一个 token 不计算
        source_len = len(input_ids[0]) - 1
        source_text = source_text.replace('<S2SV_ModStart>', '').strip()
        beam_outputs = beam_outputs.detach().cpu().tolist()
        filter_tokens = [tokenizer.eos_token, tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token,
                         tokenizer.bos_token]
        ground_truth = clean_tokens(targets[0], filter_tokens)
        language = languages[0]
        decoded_outputs = [tokenizer.decode(output[source_len:], skip_special_tokens=False) for output in beam_outputs]
        beam_outputs = [clean_tokens(output, filter_tokens) for output in decoded_outputs]
        for prediction in beam_outputs:
            if prediction.replace(' ', '') == ground_truth.replace(' ', ''):
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
        else:
            # if not correct, use the first output in the beam as the raw prediction
            raw_pred = beam_outputs[0]
            raw_predictions.append(raw_pred)
            accuracy.append(0)
        record_data.append(copy.deepcopy([source_text, ground_truth, beam_outputs, language, correct_pred]))
        nb_eval_steps += 1
        t = str(round(sum(accuracy) / len(accuracy), 4))
        bar.set_description(f"test acc: {t}")
    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")
    # write to file
    output_data = pd.DataFrame(record_data, columns=["source", "ground_truth", "predictions", "language", "correct"])
    output_dir = os.path.join(os.curdir, 'predictions')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir,
                               f'beam_{args.num_beams}_prediction.csv')
    output_data.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")

    parser.add_argument("--decoder_block_size", default=1024, type=int,
                        help="")
    parser.add_argument("--output_size", default=128, type=int,
                        help="")

    parser.add_argument("--max_stat_length", default=-1, type=int,
                        help="")

    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                        help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_pretrained_model", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    args = parser.parse_args()
    # Setup CUDA, GPU
    args.n_gpu = 1
    args.device = "cuda:0"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    # Set seed
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(
        ["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>", "<S2SV_Indent>"],
        special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    # print(model)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.load_pretrained_model:
            checkpoint_prefix = 'checkpoint-best-loss/pretrained_model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model.load_state_dict(torch.load(output_dir, map_location=args.device))
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        train(args, train_dataset, model, tokenizer, eval_dataset)
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset)



if __name__ == "__main__":
    main()
