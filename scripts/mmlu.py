"""
Adapted from https://github.com/hendrycks/test/blob/master/evaluate.py
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

choices = ["A", "B", "C", "D"]


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    choice = ["A", "B", "C", "D"]
    ans = [tokenizer(i, add_special_tokens=False).input_ids for i in choice]

    file = open(f"python/{subject}.txt", "w")

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        input_ids: torch.Tensor = tokenizer(
            prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids: torch.Tensor = tokenizer(
                prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        with torch.no_grad():
            outputs = model(input_ids,
                            output_hidden_states=True,
                            output_attentions=True)
            logits = outputs.logits

        last_token_logits: torch.Tensor = logits[0, -1, :]
        ABCD = torch.tensor([
            last_token_logits[ans[0]],
            last_token_logits[ans[1]],
            last_token_logits[ans[2]],
            last_token_logits[ans[3]],
        ])
        probs = (torch.nn.functional.softmax(
            ABCD,
            dim=0,
        ).detach().cpu().numpy())
        label = test_df.iloc[i, test_df.shape[1] - 1]
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        probs = softmax(np.array(probs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

        print(prompt, file=file)
        print(f"Size: {input_ids.numel()}", file=file)
        print(" ".join([str(i) for i in input_ids.flatten().cpu().tolist()]),
              file=file)
        print(" ".join([str(i) for i in ABCD.flatten().cpu().tolist()]),
              file=file)
        print("Model's answer: {}, expected answer: {}".format(pred, label),
              file=file)

    file.close()

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    model = args.model
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(os.path.join(args.data_dir, "test"))
        if "_test.csv" in f
    ])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    print(subjects)
    print(args)

    print(model)
    all_cors = []

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev",
                                          subject + "_dev.csv"),
                             header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test",
                                           subject + "_test.csv"),
                              header=None)

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df,
                                test_df)
        all_cors.append(cors)

        test_df["{}_correct".format(model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(model, choice)] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, "{}.csv".format(subject)),
                       index=None)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=
        "/home/scratch.trt_llm_data/llm-models/llama-models-v3/llama-v3-8b-instruct-hf/",
    )
    args = parser.parse_args()
    main(args)
