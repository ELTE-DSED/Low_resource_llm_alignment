import random
import tqdm
import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import fire

# QUESTION TYPE
# creative writing.
# 0: GENERAL (ShareGPT + Dolly)


CHATGPT_LANGUAGES = {
    "it": "Italian",
    "pl": "Polish",
    "ru": "Russian",
    "sk": "Slovak",
    "pt": "Portuguese",
    "ro": "Romanian",
    "da": "Danish",
    "sv": "Swedish",
    "no": "Norwegian",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "cs": "Czech",
    "de": "German",
    "fi": "Finnish",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "fa": "Persian",
    "hu": "Hungarian",
    "he": "Hebrew",
    "el": "Greek",
    "ar": "Arabic",
    "kr": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-traditional": "Chinese (Traditional)",
    "zh-simplified": "Chinese (Simplified)",
    "th": "Thai",
    "vi": "Vietnamese",
}


def remove_leading_fraction(input_string):
    # remove leading fraction
    cleaned_string = re.sub(r"^\d+\s*/\s*\d+", "", input_string)
    cleaned_string = re.sub(r"\d+\s*/\s*\d+$", "", cleaned_string)
    cleaned_string = cleaned_string.split("1 / 1", 1)[-1]

    # \uc9c0\uae08 \ubc88\uc5ed\ud558\uae30
    cleaned_string = cleaned_string.split("지금 번역하기")[0]

    # Language: English
    cleaned_string = cleaned_string.split("\n \n Language: ")[0]

    cleaned_string = cleaned_string.strip()

    if cleaned_string.endswith("Share Prompt"):
        cleaned_string = cleaned_string[: -len("Share Prompt")].strip()

    if cleaned_string.endswith("Translate now"):
        cleaned_string = cleaned_string[: -len("Translate now")].strip()

    for lang_code in CHATGPT_LANGUAGES:
        lang_suffix = f"Language: {CHATGPT_LANGUAGES[lang_code]}"
        if cleaned_string.endswith(lang_suffix):
            cleaned_string = cleaned_string[: -len(lang_suffix)].strip()

    # ~The following is a conversation with Bing, not ChatGPT.~
    if cleaned_string.startswith(
        "~The following is a conversation with Bing, not ChatGPT.~"
    ):
        cleaned_string = cleaned_string[
            len("~The following is a conversation with Bing, not ChatGPT.~") :
        ].strip()

    return cleaned_string


# def load_dolly_data(length_bonus):
    
#     dataset = load_dataset("databricks/databricks-dolly-15k")["train"]
#     category_to_examples = {}
#     for example in dataset:
#         category = example["category"]
#         if category not in category_to_examples:
#             category_to_examples[category] = []
#         category_to_examples[category].append(example)

#     merged_examples = []
#     for data in [
#         category_to_examples["creative_writing"],
#         category_to_examples["brainstorming"],
#         category_to_examples["open_qa"],
#         category_to_examples["general_qa"],
#         # category_to_examples["classification"],
#     ]:
#         for i in range(len(data)):
#             assert data[i]["context"] == ""
#             merged_examples.append(
#                 {
#                     "instruction": data[i]["instruction"],
#                     "input": "",
#                     "output": "",
#                     "length_bonus": length_bonus,
#                     "question_type": 0,
#                 }
#             )
#     print("Dolly examples:", len(merged_examples))
#     return merged_examples


    






def filter_and_clean_examples(merged_examples):
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b")
    max_seq_length = 256
    filtered_examples = []
    set_of_unique_instructions = set()

    # Filter out examples with non-ascci characters
    merged_examples = [
        {
            "instruction": remove_leading_fraction(example["instruction"]),
            "input": example.get("input",""),
            "output": "",
            "length_bonus": example["length_bonus"],
            "question_type": example["question_type"],
        }
        for example in merged_examples
    ]
    

    for example in tqdm.tqdm(merged_examples):
        instruction = example["instruction"]
        instruction_token_length = len(tokenizer.encode(instruction))
        if (
            2 <= instruction_token_length <= max_seq_length
            and instruction not in set_of_unique_instructions
        ):
            filtered_examples.append(example)
            set_of_unique_instructions.add(instruction)

    return filtered_examples

    


def load_json(share_gpt_data_path):
    with open(share_gpt_data_path, "r") as f:
        share_gpt_data = json.load(f)
    examples = []
    for data in share_gpt_data:
        examples.append(
            {
                "instruction": data["instruction"],
                "input": "",
                "output": "",
            }
        )
    print(f"{share_gpt_data_path} examples:", len(examples))
    return examples










def alpaca_scheduling_samples(length_bonus, seed=42, json_path=None, 
                              sampling_probs=None, mode="balanced",
                              max_samples=None):
    """
    Build a balanced or weighted Alpaca sample list, optionally limited in size.

    Args:
        length_bonus (float): bonus length to assign to examples
        seed (int): random seed
        json_path (str): path to the JSON file
        sampling_probs (dict): optional dict of {structural_class: probability}
                               used only if mode="weighted"
        mode (str): "balanced" (round-robin) or "weighted" (probabilistic)
        max_samples (int or None): max number of samples to return

    Returns:
        List[dict]: list of sample dicts ready for RL
    """
    if json_path is None:
        json_path = "data/sampled_alpaca_level1_table_sublist_with_ids_with_prose.json"

    # Load JSON
    df = pd.read_json(json_path)
    print(f"Loaded {len(df)} rows from {json_path}")

    # Rename 'class' to 'structural_class' if needed
    if "class" in df.columns and "structural_class" not in df.columns:
        df = df.rename(columns={"class": "structural_class"})

    required_cols = {"instruction", "input", "structural_class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"JSON file is missing required columns: {missing}")

    df = df.drop_duplicates()
    print(f"Total rows after dedup: {len(df)}")

    # -----------------------------------
    # Exclude unwanted classes
    # -----------------------------------
    df = df[~df["structural_class"].str.lower().str.contains("table")]
    if df.empty:
        raise ValueError("No samples left after filtering structural classes.")

    # -----------------------------------
    # balanced round-robin balancing
    # -----------------------------------
    if mode == "balanced":
        # Balance per class by downsampling to min_count
        min_count = df["structural_class"].value_counts().min()
        balanced_df = (
            df.groupby("structural_class", group_keys=False)
              .apply(lambda x: x.sample(min_count, random_state=seed))
              .reset_index(drop=True)
        )

        # Shuffle inside each class
        grouped = {
            cls: group.sample(frac=1, random_state=seed).reset_index(drop=True)
            for cls, group in balanced_df.groupby("structural_class")
        }
        classes = list(grouped.keys())
        random.Random(seed).shuffle(classes)

        # Interleave strictly round-robin
        interleaved_rows = []
        for rows in zip(*[grouped[cls].itertuples(index=False) for cls in classes]):
            interleaved_rows.extend(rows)

        # Apply max_samples limit
        if max_samples is not None:
            interleaved_rows = interleaved_rows[:max_samples]

    # -----------------------------------
    # Weighted sampling using probabilities
    # -----------------------------------
    elif mode == "weighted":
        if sampling_probs is None:
            raise ValueError("sampling_probs must be provided in weighted mode.")

        # Normalize probabilities
        total_prob = sum(sampling_probs.values())
        probs = {k: v / total_prob for k, v in sampling_probs.items()}

        # Group by class
        grouped = {cls: group for cls, group in df.groupby("structural_class") if cls in probs}
        if not grouped:
            raise ValueError("No structural classes match the provided sampling_probs.")

        rng = np.random.default_rng(seed)

        # Determine total number of samples to draw
        total_samples = sum(len(g) for g in grouped.values())
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)

        classes_list = list(grouped.keys())
        class_sizes = {cls: len(g) for cls, g in grouped.items()}

        # Sample classes based on probabilities
        chosen_classes = rng.choice(classes_list, size=total_samples, p=[probs[c] for c in classes_list])

        # Track indexes for each class
        class_indices = {cls: 0 for cls in classes_list}
        interleaved_rows = []
        for cls in chosen_classes:
            idx = class_indices[cls]
            if idx < class_sizes[cls]:
                interleaved_rows.append(grouped[cls].iloc[idx])
                class_indices[cls] += 1
            # If class exhausted, skip

    else:
        raise ValueError("mode must be 'balanced' or 'weighted'")

    # -----------------------------------
    # Build final examples
    # -----------------------------------
    merged_examples = [
        {
            "instruction": row.instruction,
            "input": row.input,
            "output": "",
            "length_bonus": length_bonus,
            "question_type": 1,
        }
        for row in interleaved_rows
    ]

    print(f"Total generated examples ({mode} mode): {len(merged_examples)}")
    return merged_examples




















# def alpaca_prose(length_bonus, seed=42):
    
#     csv_path = "data/Checkpoint_350_alpaca_preference_structural_rule_based_only_prose_structures_withids_output_merge.csv"
    
#     df = pd.read_csv(csv_path)
    
#     if df.empty:
#         raise ValueError("No samples left after filtering structural classes.")
    
#     merged_examples = [
#         {
#             "instruction": row.instruction,
#             "input": row.input,
#             "output": "",
#             "length_bonus": length_bonus,
#             "question_type": 1,
#         }
#         for row in df.itertuples()
#     ]
    
    
#     print(f"Balanced Alpaca examples: {len(merged_examples)}")
    
#     return merged_examples[:500];
















def load_Nvidia_steer_helpfulness2_data(length_bonus):
    # Load the dataset from a JSON file
    dataset = load_dataset("nvidia/HelpSteer2")["train"]
    
    merged_examples = []
    for example in dataset:
        merged_examples.append(
            {
                "instruction": example["prompt"],  # adjust this if your JSON field name is different
                "input": "",
                "output": "",
                "length_bonus": length_bonus,
                "question_type": 0,
            }
        )
    
    print("HELP steer2 ask examples:", len(merged_examples))
    return merged_examples




import pandas as pd



def main(
    sharegpt_prompt_path: str = "data/RL_data/writing_share_gpt.json",
    output_file: str = "/singularity/100-gpu01/arafat_data/distillation_project/data/RL_data/self_alpaca_non_prose.json",
    general_length_bonus: float = 1.0,
    red_teaming_length_penalty: float = 1.5, 
    do_not_answer_penalty: float = -0.5, 
):


    alpaca = alpaca_scheduling_samples(general_length_bonus)

    merged_examples = []
    merged_examples.extend(alpaca)      
    
    filtered_examples = filter_and_clean_examples(merged_examples)

    print("Total examples:", len(filtered_examples))

    with open(output_file, "w") as f:
        json.dump(filtered_examples, f, indent=2);





if __name__ == "__main__":
    fire.Fire(main)