import re
import random
import argparse
import os
import math
import json
import time
import torch
import vllm
import evaluate
from string import ascii_uppercase
import pandas as pd
from tqdm import tqdm



exact_match = evaluate.load("exact_match")


###example prompt for MCQ datasets
exi = lambda q, choices, thought, answer: f"""Question: {q}
Options: {choices}

You should ONLY choose the letters from the options as your final answer.
Response: Let's think step by step. {thought} So the answer is {answer}.
"""
### from MMLU
# socialology
ex1 = exi("Mass-society theory suggests that:", "A. the content of the media is determined by market forces\nB. the subordinate classes are dominated by the ideology of the ruling class\nC. the media manipulate 'the masses' as vulnerable, passive consumers\nD. audiences make selective interpretations of media messages\n", "Mass-society theory suggests that media content is used to manipulate the masses as passive consumers, who are vulnerable to external influence. Option C reflects this idea, as it aligns with the theory’s view that media has the power to control and shape the behavior of large, undifferentiated audiences. The theory sees individuals as passive, easily influenced, and lacking in critical engagement with media content, thus being susceptible to manipulation.", "C")
# world fact
ex2 = exi("What was GDP per capita in the United States in 1850 when adjusting for inflation and PPP in 2011 prices?", "A. About $300\nB. About $3k\nC. About $8k\nD. About $15k\n", "To estimate GDP per capita in 1850 using inflation-adjusted and PPP-adjusted 2011 prices, historical economic data suggests that early industrial societies like the United States had modest per capita income compared to modern standards. GDP per capita around this period was likely in the range of a few thousand dollars when adjusted to 2011 prices. Option B, “About $3k,” aligns with historical estimates of the U.S. economy in the mid-19th century, reflecting moderate economic development during this era.", "B")
# public relations
ex3 = exi('Which common public relations tactic involves sending journalists on visits to appropriate locations?', 'A. Media release\nB. Media tour\nC. Press room\nD.Promotional days/weeks\n', "A media tour involves sending journalists to relevant locations to give them firsthand experience of a product, service, or event. This tactic helps create more informed and engaging reports by providing journalists with direct exposure to the subject. Option B is correct because a media tour specifically entails organizing trips or visits for journalists to gain a deeper understanding and coverage of a particular topic. Other options, like media releases, do not involve physical visits.", 'B')
# electrical engineering
ex4 = exi('Potentiometer method of DC voltage measurement is more accurate than direct measurement using a voltmeter because', 'A. It loads the circuit moderately.\nB. It loads the circuit to maximum extent.\nC. It uses centre zero galvanometer instead of voltmeter.\nD. It does not load the circuit at all.\n', '', 'D')
# business ethics
ex5 = exi('What does Milton Friedman believe to be the sole responsibility of business?', 'A. The only social responsibility of business is to its shareholders\nB. Managers should act in ways that balance the interest of society and shareholders\nC. The primary responsibility organizations have is to its employees\nD. The primary responsibility organizations have is to its stakeholders\n', '', 'A')
### few-shot
generic_prompt = lambda q, choices: f"""
{ex1}

{ex2}

{ex3}

Question: {q}
Options: {choices}

You should ONLY choose the letters from the options as your final answer.
Response: Let's think step by step. """


def extract_answer(text):
    # The regular expression pattern
    pattern = r"(?:the\s*(?:correct)?\s*answer\s*is\s*\(?([A-Za-z])\)?\s*)"
    pattern2 = r'Option\s+([A-Za-z])\s+is\s+(?:the\s+)?correct(?:\s+answer)?'
    pattern3 = r'(?:</hCoT>|Option\s+)([A-Za-z])(?:\s*,.*?)?\sis\s+(?:the\s+)?correct(?:\s+answer)?'

    # Find all matches in the text
    matches = re.findall(pattern, text, re.IGNORECASE)
    matches2 = re.search(pattern2, text, re.IGNORECASE)
    matches3 = re.search(pattern3, text, re.IGNORECASE)
    # Return the last match if any, or None
    if matches:
        return matches[-1]
    elif matches2:
        return matches2.group(1)
    elif matches3:
        return matches3.group(1)
    else:
        return None
    

def main(args):
    random.seed(42)

    print("Loading data...")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    if args.eval_dataset == 'arc-challenge':
        letter_mapping = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
        nums = list(letter_mapping.keys())

        all_data = []
        with open(os.path.join(args.data_dir, "test.jsonl")) as f:
            for line in f.readlines():
                all_data.append(json.loads(line))

        if args.num_instances is not None and len(all_data) >= args.num_instances:
            all_data = all_data.sample(args.num_instances, random_state=42)

        prompts = []
        targets = []
        for data in all_data:
            question = data['question']['stem'].strip()

            choices = ''
            for choice in data['question']['choices']:
                choice_label = choice['label'] if choice['label'] not in nums else letter_mapping[choice['label']]
                choices += f"{choice_label}. {choice['text']}\n"

            prompt = generic_prompt(question, choices)
            prompts.append(prompt)
            
            answerKey = data['answerKey'] if data['answerKey'] not in nums else letter_mapping[data['answerKey']]
            targets.append(answerKey)

    elif args.eval_dataset == 'csqa':
        id_to_option = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E"
        }
        prompts = []
        targets = []
        processed_data = []
        with open(os.path.join(args.data_dir, f"dev.jsonl")) as fin:
            for line in fin:
                example = json.loads(line)
                processed_data.append(example)

        if args.num_instances is not None and len(processed_data) >= args.num_instances:
            processed_data = random.sample(processed_data, args.num_instances)

        for example in processed_data:
            choices = ''
            for i, option in enumerate([option["text"] for option in example["question"]["choices"]]):
                choices += f"{id_to_option[i]}. {option}\n"
            prompt = generic_prompt(example['question']['stem'].strip(), choices)

            prompts.append(prompt)
            targets.append(example["answerKey"])

    elif args.eval_dataset == 'gsm8k':
        test_data = []
        with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
            for line in fin:
                example = json.loads(line)
                test_data.append({
                    "question": example["question"],
                    "answer": example["answer"].split("####")[1].strip()
                })
            
        # some numbers are in the `x,xxx` format, and we want to remove the comma
        for example in test_data:
            example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
            assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

        if args.num_instances and len(test_data) >= args.num_instances:
            test_data = random.sample(test_data, args.num_instances)

        prompt_prefix = "Answer the following question.\n\n"
        prompts = [prompt_prefix + "Question: " + example["question"].strip() + "\nAnswer: Let's think step by step." for example in test_data]
        targets = [example["answer"] for example in test_data]

    elif args.eval_dataset == 'halueval':
        all_data = []
        with open(os.path.join(args.data_dir, "test.json")) as f:
            for line in f.readlines():
                all_data.append(json.loads(line))

        if args.num_instances and len(all_data) >= args.num_instances:
            all_data = random.sample(all_data, args.num_instances)

        prompts = []
        targets = []
        for data in all_data:
            right_answer = data['right_answer']
            hallucinated_answer = data['hallucinated_answer']

            contain_hallucination = random.choice([True, False]) 

            if contain_hallucination:
                targets.append("A")
                question = f"{data['question']}\nDoes the following provided answer contains hallucination for the question based on the world knowledge?\nAnswer: {hallucinated_answer}"
            else:
                targets.append("B")
                question = f"{data['question']}\nDoes the following provided answer contains hallucination for the question based on the world knowledge?\nAnswer: {right_answer}"

            choices = ''
            for i, option in enumerate(["Yes", "No"]):
                choices += f"{ascii_uppercase[i]}. {option}\n"
            prompt = generic_prompt(question.strip(), choices)
            prompts.append(prompt)

    elif args.eval_dataset == 'strategyqa':
        option_map = {
            True: "A",
            False: "B"
        }
        with open(os.path.join(args.data_dir, "test.json")) as f:
            all_data = json.load(f)

        if args.num_instances and len(all_data) >= args.num_instances:
            all_data = random.sample(all_data, args.num_instances)
    
        prompts = []
        targets = []
        for data in all_data:
            answer = data['answer']
            targets.append(option_map[answer])

            choices = ''
            for i, option in enumerate(["Yes", "No"]):
                choices += f"{ascii_uppercase[i]}. {option}\n"
            prompt = generic_prompt(data['question'].strip(), choices)

            prompts.append(prompt)

    elif args.eval_dataset == 'truthfulqa':
        def split_multi_answer(ans, sep=';', close=True):
            """Splits string of all reference answers into a list of formatted answers"""
            answers = ans.strip().split(sep)
            split_answers = []
            for a in answers:
                a = a.strip()
                if len(a):
                    if close:  # add a period after all answers
                        if a[-1] != '.':
                            split_answers.append(a + '.')
                        else:
                            split_answers.append(a)
                    else:
                        split_answers.append(a)

            return split_answers

        BEST_COL = 'Best Answer'
        ANSWER_COL = 'Correct Answers'
        INCORRECT_COL = 'Incorrect Answers'

        questions = pd.read_csv(os.path.join(args.data_dir, "TruthfulQA.csv"))

        if args.num_instances is not None and len(questions) >= args.num_instances:
            questions = questions.sample(args.num_instances, random_state=42)

        prompts = []
        targets = []
        for idx in questions.index:
            # reference answers
            ref_best = questions.loc[idx, BEST_COL].strip()
            if ref_best[-1] != ".":
                ref_best += "."
            ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

            options = [ref_best] + ref_false
            random.shuffle(options)

            targets.append(ascii_uppercase[options.index(ref_best)])

            choices = ''
            for i, option in enumerate(options):
                choices += f"{ascii_uppercase[i]}. {option}\n"
            prompt = generic_prompt(questions.loc[idx]['Question'].strip(), choices)
            prompts.append(prompt)

    else:
        raise ValueError('The eval_dataset is currently not supported!')

    print("Loading model and tokenizer...")

    model = vllm.LLM(
        model=args.model_name_or_path,
        tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
        tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
        tensor_parallel_size=torch.cuda.device_count(),
        tokenizer_revision=args.hf_revision,
        revision=args.hf_revision,
    )
    stop_strings = args.additional_stop_sequence
    if args.newline_stop:
        if args.stop_at_double_newline:
            stop_strings += ["\n\n"] 
        elif args.stop_at_triple_newline:
            stop_strings += ["\n\n\n"] 
        else:
            stop_strings += ["\n"]
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=args.clm_max_length,
        stop=stop_strings,
        skip_special_tokens=False,
    )
    
    # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {
        g.prompt: g.outputs[0].text for g in generations
    }
    outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]


    print("Calculating accuracy...")
    predictions = []
    for output in outputs:
        answer = extract_answer(output)
        if answer:
            predictions.append(answer)
        else:
            predictions.append("")
        
    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")


    predictions = [{
        "prompt": prompt,
        "answer": tgt,
        "model_output": output,
        "prediction": pred
    } for prompt, tgt, output, pred in zip(prompts, targets, outputs, predictions)]

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        help="The HuggingFace model to be evaluated."
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="if specified, we will load the model from a revision of the model in the hub"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="If specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--data_dir", 
        type=str,
    )
    parser.add_argument(
        "--save_dir", 
        type=str
    )
    parser.add_argument(
        "--num_instances", 
        type=int, 
        default=None, 
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
    parser.add_argument(
        "--newline_stop",
        action="store_true",
        help="If given, we will use stop token (usually newline or double newline) to stop generation."
    )
    parser.add_argument(
        "--stop_at_double_newline",
        action="store_true",
        help="If given, will stop generation at double newline instead of single."
    )
    parser.add_argument(
        "--stop_at_triple_newline",
        action="store_true",
        help="If given, will stop generation at triple newline instead of single."
    )
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
    )
    parser.add_argument(
        "--clm_max_length",
        type=int,
        default=256
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        choices=['arc-challenge', 'csqa', 'gsm8k', 'halueval', 'strategyqa', 'truthfulqa'],
        default='arc-challenge'
    )


    args = parser.parse_args()
    main(args)