import argparse
import json
from transformers import AutoTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_dir', type=str, help="The name or path of tokenizer", required=True)
    parser.add_argument('--input_json', type=str, help="The path of input json file", required=True)
    parser.add_argument('--output_json', type=str, help="The path of output json file", required=False)
    parser.add_argument('--mode', type=str, choices=["encode", "decode"], required=True)
    args = parser.parse_args()
    return args

def encode(args, tokenizer):
    f = open(args.input_json)
    input_data = json.load(f)
    print("input_str successfully loaded.")
    print(input_data)
    tokenizer_output = tokenizer(input_data["input"], padding=True, return_tensors="pt")
    position_ids = tokenizer_output["attention_mask"].cumsum(-1) - 1
    output_dict = {
        "input_ids": tokenizer_output["input_ids"].numpy().tolist(),
        "attention_mask": tokenizer_output["attention_mask"].numpy().tolist(),
        "position_ids": position_ids.numpy().tolist()
    }

    with open(args.output_json, "w") as output_file:
        json.dump(output_dict, output_file)

    print(f"Dumped the below info into {args.output_json}")
    print(output_dict)


def decode(args, tokenizer):
    f = open(args.input_json)
    input_data = json.load(f)
    print("input_ids successfully loaded.")
    print(input_data)
    output = tokenizer.batch_decode(input_data["input_ids"], skip_special_tokens=True)
    print("Decoded string is:", output)

def main():
    args = parse_arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if args.mode == "encode":
        assert args.output_json, "output_json needs to be present for encode mode"
        encode(args, tokenizer)
    else:
        decode(args, tokenizer)

if __name__ == '__main__':
    main()