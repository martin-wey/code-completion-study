import logging
import argparse
import sys
import difflib

from pprint import pprint
from tqdm import tqdm

import torch
from torch.utils.data import SequentialSampler, DataLoader

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from ast_parser.markers import SPECIAL_MARKERS, SPECIAL_TAGS, SYNTAX_TOKENS

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/CodeGPT-small-java',
                        help='Path to the model checkpoint or its name in Huggingface hub')
    parser.add_argument('--eval_path', type=str, default=None,
                        help='Path to the evaluation file.')
    parser.add_argument('--code_flows_path', type=str, default=None,
                        help='Code flows corresponding to the evaluation file.')
    parser.add_argument('--all_tokens', type=bool, default=True,
                        help='Whether to predict all tokens or only identifiers.')
    parser.add_argument('--add_bos_eos', type=bool, default=True,
                        help='Whether to add bos and eos token on each sample.')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    data_files = {'test': args.eval_path,
                  'flows': args.code_flows_path}

    eval_dataset = load_dataset('text', data_files=data_files)

    def tokenize_function(samples):
        if args.add_bos_eos:
            samples['text'] = ['<s> ' + line + ' </s>' for line in samples['text'] if
                               len(line) > 0 and not line.isspace()]
        else:
            samples['text'] = [line for line in samples['text'] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            samples['text'],
            padding=False,
            truncation=True,
            max_length=1024,
            return_attention_mask=False
        )

    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True
    )
    tokenized_eval_dataset.set_format(type='torch', columns=['input_ids'])
    logger.info(f'Evaluation set: {tokenized_eval_dataset}')

    eval_sampler = SequentialSampler(tokenized_eval_dataset['test'])
    eval_dataloader = DataLoader(tokenized_eval_dataset['test'], sampler=eval_sampler)

    model.eval()

    total_pred = []
    total_gt = []

    if args.all_tokens:
        acc = 0.0
        ratio = 0.0
        n_test = 0
    else:
        acc_per_token_type = {'any': {'acc': 0.0, 'n_test': 0, 'ratio': 0}}
        tags = SPECIAL_TAGS + ['[ID]']
        for tag in tags:
            acc_per_token_type[tag] = {
                'acc': 0.0,
                'n_test': 0,
                'ratio': 0
            }

    # similar to https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/CodeCompletion-token/code/run_lm.py
    #   adapted for our situation where we want to compute the accuracy per type of token,
    #   or the accuracy for all tokens or only syntactical tokens.
    for step, sample in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            logits = model(sample['input_ids'], labels=sample['input_ids'])
            pred_ids = logits[1].argmax(-1)
            # loss += logits[0].item()

        all_pred = []
        all_gt = []
        for pred, gt in zip(pred_ids, sample['input_ids']):
            pred = pred.cpu().tolist()
            gt = gt.cpu().tolist()

            now_gt = None
            now_pred = None
            for i, y in enumerate(gt):
                if i == 0:
                    # predict bos token
                    current_gt = [y]
                    current_pred = [0]
                    now_gt = [y]
                    now_pred = [0]
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                             tokenizer.pad_token_id]:
                        all_pred.append(tokenizer.decode(current_pred).strip().split()[0])
                        all_gt.append(tokenizer.decode(current_gt).strip())
                        now_gt.clear()
                        now_pred.clear()
                else:
                    # \u0120 == space = beginning/end of a token
                    if tokenizer.convert_ids_to_tokens(y)[0] == '\u0120':
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append('<SPACE>')
                            all_gt.append(tokenizer.decode(now_gt).strip())
                            now_gt.clear()
                            now_pred.clear()
                    if y in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id,
                             tokenizer.pad_token_id]:
                        if len(now_gt) > 0:
                            try:
                                all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                            except IndexError:
                                all_pred.append('<SPACE>')
                            all_gt.append(tokenizer.decode(now_gt).strip())
                        now_gt = [y]
                        now_pred = [pred[i - 1]]
                        try:
                            all_pred.append(tokenizer.decode(now_pred).strip().split()[0])
                        except IndexError:
                            all_pred.append('<SPACE>')
                        all_gt.append(tokenizer.decode(now_gt).strip())
                        now_gt.clear()
                        now_pred.clear()
                        continue
                now_gt.append(y)
                now_pred.append(pred[i - 1])
        assert len(all_pred) == len(all_gt)

        total_pred.extend(all_pred)
        total_gt.extend(all_gt)

        prev_type = []
        prev_idx_flow = 0
        for x, y in zip(all_pred, all_gt):
            code_flow = tokenized_eval_dataset['flows']['text'][step].split()
            if y not in ['<s>', '</s>', '<EOL>', '<pad>'] and y in SYNTAX_TOKENS:
                if args.all_tokens:
                    if x == y:
                        acc += 1
                    ratio += difflib.SequenceMatcher(None, x, y).ratio() * 100
                    n_test += 1
                else:
                    # specific prediction for types
                    try:
                        if y in code_flow and y not in SYNTAX_TOKENS:
                            # index of the current identifier in the code flow
                            current_idx = code_flow.index(y, prev_idx_flow)

                            for i, token in enumerate(code_flow[prev_idx_flow:current_idx]):
                                if token in SPECIAL_TAGS:
                                    prev_type = [token]
                                if token == SPECIAL_MARKERS['identifier']:
                                    if prev_type[-1] in [SPECIAL_MARKERS['function'], SPECIAL_MARKERS['parameter']]:
                                        prev_type = [token]
                                    elif token not in prev_type:
                                        prev_type.append(token)

                            prev_idx_flow = current_idx + 1

                            if x == y:
                                acc_per_token_type[prev_type[0]]['acc'] += 1
                                acc_per_token_type['any']['acc'] += 1
                            sim_ratio = difflib.SequenceMatcher(None, x, y).ratio() * 100
                            acc_per_token_type[prev_type[0]]['ratio'] += sim_ratio
                            acc_per_token_type['any']['ratio'] += sim_ratio
                            acc_per_token_type[prev_type[0]]['n_test'] += 1
                            acc_per_token_type['any']['n_test'] += 1
                    except:
                        # we might get an exception if the identifier that was predicted
                        #   does not appear in the code flow.
                        continue

    if args.all_tokens:
        if n_test != 0:
            acc = round(acc / n_test, 4)
            ratio = round(ratio / n_test, 2)
        print(acc, ratio, n_test)
    else:
        for type in acc_per_token_type:
            if acc_per_token_type[type]['n_test'] != 0:
                acc_per_token_type[type]['acc'] = round(acc_per_token_type[type]['acc'] /
                                                        acc_per_token_type[type]['n_test'], 4)
                acc_per_token_type[type]['ratio'] = round(acc_per_token_type[type]['ratio'] /
                                                          acc_per_token_type[type]['n_test'], 2)

        pprint(acc_per_token_type)


if __name__ == '__main__':
    main()
