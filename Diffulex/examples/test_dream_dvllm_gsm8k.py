import os
import csv
import time
import argparse

import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from viztracer import VizTracer
from transformers import AutoTokenizer

from d2f_engine import LLM, SamplingParams


def summarize_profiling(csv_path: str) -> dict:
    totals = {}
    total_nums = {}
    avgs = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    val = float(v)
                except ValueError:
                    continue
                if val != 0.0:
                    total_nums[k] = total_nums.get(k, 0) + 1
                totals[k] = totals.get(k, 0.0) + val
    print(pd.DataFrame([totals]).T)
    for k, v in totals.items():
        if k in total_nums and total_nums[k] > 0:
            avgs[k] = v / total_nums[k]
        else:
            avgs[k] = 0.0
    print(pd.DataFrame([avgs]).T)
    

FEW_SHOTS="""
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Test Dream DVLLM on GSM8K dataset')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the base model (e.g., Dream-org/Dream-v0-Base-7B)')
    parser.add_argument('--lora-path', type=str, default=None,
                        help='Path to LoRA weights (optional)')
    parser.add_argument('--use-lora', action='store_true',
                        help='Enable LoRA')
    parser.add_argument('--dataset', type=str, default="openai/gsm8k",
                        help='Path to GSM8K dataset (default: openai/gsm8k from HuggingFace)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to test (default: all)')
    parser.add_argument('--data-parallel-size', type=int, default=1,
                        help='Data parallel size (default: 1)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Tensor parallel size (default: 1)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='GPU memory utilization (default: 0.9)')
    parser.add_argument('--max-num-seqs', type=int, default=20,
                        help='Max number of sequences (default: 20)')
    parser.add_argument('--max-model-len', type=int, default=2048,
                        help='Max model length (default: 2048)')
    parser.add_argument('--kv-cache-layout', type=str, default="unified",
                        choices=["unified", "distinct"],
                        help='KV cache layout (default: unified)')
    parser.add_argument('--output-file', type=str, default="log/profiles/perf_dvllm_dream_7B.json",
                        help='Output profiling file')
    parser.add_argument('--skip-sleep', action='store_true',
                        help='Skip the 60 second sleep before generation')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=256,
                        help='Max tokens to generate (default: 256)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate model path
    if not os.path.exists(args.model) and not args.model.count('/') == 1:
        print(f"Error: Model path '{args.model}' does not exist.")
        print("Please provide a valid local path or HuggingFace model name (org/model).")
        exit(1)

    # Validate LoRA path if provided
    if args.lora_path and not os.path.exists(args.lora_path) and not args.lora_path.count('/') == 1:
        print(f"Error: LoRA path '{args.lora_path}' does not exist.")
        print("Please provide a valid local path or HuggingFace model name (org/model).")
        exit(1)

    print(f"Initializing LLM with model: {args.model}")
    if args.use_lora and args.lora_path:
        print(f"Using LoRA weights from: {args.lora_path}")

    try:
        llm = LLM(
            args.model,
            lora_path=args.lora_path if args.use_lora else None,
            use_lora=args.use_lora,
            model_name="dream",
            model_type="diffusion_lm",
            enforce_eager=True,
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=2048,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
            accept_threshold=0.95,
            complete_threshold=0.9,
            add_new_block_threshold=0.1,
            kv_cache_layout=args.kv_cache_layout
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        print("\nNote: This script requires a CUDA-enabled GPU to run.")
        exit(1)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    print(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset, "main")['test']['question'][:]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Note: If the dataset is not in HuggingFace cache, it will try to download it.")
        exit(1)

    # Limit samples if specified
    if args.num_samples:
        dataset = dataset[:args.num_samples]
        print(f"Testing on {args.num_samples} samples")
    else:
        print(f"Testing on {len(dataset)} samples")

    print("Preparing prompts...")
    prompts = [tokenizer.bos_token + FEW_SHOTS + p for p in tqdm(dataset)]

    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    # Optional sleep before generation
    if not args.skip_sleep:
        print("Sleeping for 60 seconds before generation...")
        time.sleep(60)

    print("Starting generation...")
    s = time.time()
    outputs = llm.generate(prompts, sampling_params)
    e = time.time()

    print("=*=" * 30)
    print("\nProfiling Results\n")
    print("=*=" * 30)
    print(f"Generated {len(outputs)} outputs.")
    print(f"Total tokens: {sum(len(o['token_ids']) for o in outputs)}")
    print(f"Total time: {e - s:.2f} seconds.")
    print(f"Avg TPS: {sum(len(o['token_ids']) for o in outputs) / (e - s):.2f} tok/s.")
    print(f"AVG Number of Diffusion Steps: {sum(o['n_diff_steps'] for o in outputs) / len(outputs):.2f}")
    print("=*=" * 30)

    for idx, o in enumerate(outputs):
        print("\n", "=*=" * 30)
        print(f"[Prompt {idx} Result] \n{prompts[idx] + '\n-----<Start-of-Response>-----\n' + o['text']}\n")