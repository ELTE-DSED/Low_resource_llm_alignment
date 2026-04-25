from argparse import Namespace
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import tqdm
import json
import random
import pandas as pd
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import fire
import os
from collections import defaultdict
from models.qlora_model import load_4bit_model_for_inference


from rule_based_contrastive_sampling.utils import *



HF_TOKEN = os.getenv("HF_TOKEN")
print(HF_TOKEN)
login(token=HF_TOKEN);



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
);



positive_aligning_prompt = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.

{RuleBasedInfo}

### Instruction:
{Input}{Context}
### Response:{RevealedOutput}
"""

negative_aligning_prompt = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.

{RuleBasedInfo}

### Instruction:
{Input}{Context}
### Response:
"""



def _parse_indices_helper(indices):
    for index in indices.split(','):
        if '-' in index:
            start, end = index.split('-')
            for i in range(int(start), int(end) + 1):
                yield i
        else:
            yield int(index)


def parse_indices(indices):
    return list(_parse_indices_helper(indices))


def process_filter_result(result):
    if not result or result == "":
        return False, None

    if "#" in result:
        result = result[:result.index('#')].strip()

    return True, result




def generate_result(model, tokenizer, prompts, max_new_tokens=256):
    
    if type(prompts) == str:
        prompts = [prompts];
        
        
    bad_words = ['\\', '\\\\', '`', '```']
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    
    all_results = [None for _ in range(len(prompts))]
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda');
        
        results = model.generate(**inputs, temperature=0.7, do_sample=True, max_new_tokens=max_new_tokens);
        
        for i in range(len(prompts)):
            result = results[i][len(inputs['input_ids'][i]):]
            result = tokenizer.decode(result, skip_special_tokens=True)
            status, result = process_filter_result(result)
            all_results[i] = (status, result)
    return all_results




# NEW: Function to calculate distance based on structural class
def calculate_distance(text, reference_text, structural_class):
    """
    Calculate distance between generated text and reference based on structural class.
    Returns distance score between 0 and 1.
    """
    if structural_class == "Prose":
        
        generated_list_distribution_comma, generated_list_distribution_dashed_numbered, generated_question_distribution, generated_number_distribution, generated_dialogue_distribution, _, _ = prose_distribution_modeling(text)
        reference_list_distribution_comma, reference_list_distribution_dashed_numbered, reference_question_distribution, reference_number_distribution, reference_dialogue_distribution, _ , _ = prose_distribution_modeling(reference_text);
        
        return (structure_distance(generated_list_distribution_comma, reference_list_distribution_comma, include_absolute_length = False) + structure_distance(generated_list_distribution_dashed_numbered, reference_list_distribution_dashed_numbered, include_absolute_length = False) + structure_distance(generated_question_distribution, reference_question_distribution, include_absolute_length = False) + structure_distance(generated_number_distribution, reference_number_distribution, include_absolute_length = False) +   structure_distance(generated_dialogue_distribution, reference_dialogue_distribution, include_absolute_length = False)) / 5
        
    if structural_class == "Mathematics":
        generated_dist = maths_distribution_modeling(text)
        reference_dist = maths_distribution_modeling(reference_text)
        return structure_distance(generated_dist, reference_dist)
    
    elif structural_class == "Code":
        generated_dist = code_distribution_modeling(text)
        reference_dist = code_distribution_modeling(reference_text)
        return structure_distance(generated_dist, reference_dist)
    
    elif structural_class == "Dialog":
        generated_dist = dialogue_distribution_modeling(text)
        reference_dist = dialogue_distribution_modeling(reference_text)
        return structure_distance(generated_dist, reference_dist)
    
    elif structural_class == "Questions":
        generated_dist = question_distribution_modeling(text)
        reference_dist = question_distribution_modeling(reference_text)
        return structure_distance(generated_dist, reference_dist)
    
    elif structural_class == "Number":
        generated_dist = number_distribution_modeling(text)
        reference_dist = number_distribution_modeling(reference_text)
        return structure_distance(generated_dist, reference_dist)
    
    else:
        return 0.0


def _compute_margin(dist_fn, full_output, positive_sample, text, input_context, structural_distribution_weight, tokens_absolute_length_weight, output_variance_weight, input_ground_overlap_weight, max_distance):
    ref_dist      = dist_fn(full_output)
    positive_dist = dist_fn(positive_sample)
    neg_dist      = dist_fn(text)

    pos_structural_sim = structure_distance(ref_dist, positive_dist)
    neg_structural_sim = structure_distance(ref_dist, neg_dist)

    pos_distance = (structural_distribution_weight*pos_structural_sim + input_ground_overlap_weight*input_ground_overlap_distance(input_context, full_output, positive_sample) + 
               output_variance_weight*output_variance_distance(full_output, positive_sample) + tokens_absolute_length_weight*tokens_absolute_length_difference(positive_sample, full_output));
    
    neg_distance = (structural_distribution_weight*neg_structural_sim + input_ground_overlap_weight*input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output))

    margin = neg_distance - pos_distance

    print(f"pos_distance {pos_distance}, neg_distance {neg_distance}");
    
    return margin, margin < -min_margin, pos_distance, neg_distance




def validate_rule_based_info(text, rule_based_info, structural_class, is_positive=True,
                              revealed_output="", full_output="", positive_sample="",
                              max_distance_threshold=0.7, min_distance_threshold=0.9,
                              input_context="", input_only = "", instruction_only = "", structural_distribution_weight = 0.7 ,
                                tokens_absolute_length_weight = 0.1,output_variance_weight = 0.1,input_ground_overlap_weight = 0.1,                                
                                max_distance = 0.25):

    
    # has_code            = is_code_present(text)
    # has_math            = is_math_present(text)
    # has_list            = is_list_present(text)
    # has_single_number   = detect_single_number(text)
    # has_dialog          = is_dialog(text)
    # has_question        = is_question(text)
    has_table           = detect_table(text)
    # has_numbered_list   = detect_numbered_list(text);
    # has_comma_list      = detect_comma_list(text);
    # has_dashed_list     = detect_dashed_list(text)
    
    # # has_any_structure = has_code or has_math or has_list or has_question
    # assert structural_cues_weights != None;
    
    # if structural_cues_weights:
    #     structural_distribution_weight = structural_cues_weights['structural_cues_weights'];
    #     tokens_absolute_length_weight = structural_cues_weights['tokens_absolute_length'];
    #     output_variance_weight = structural_cues_weights['output_variance'];
    #     input_ground_overlap_weight = structural_cues_weights['input_ground_overlap'];

        
    if is_positive:
        
        for rule in rule_based_info:
            
            if structural_class == "Prose":
                structural_distance = calculate_distance(text, full_output, structural_class);
                distance = (structural_distribution_weight * structural_distance + input_ground_overlap_weight * input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output));
                
                if distance > max_distance_threshold or (text == instruction_only) or (text == input_only):
                    return False, False,distance,1;
                
            if "table" in rule.lower():
                if not has_table or (text == instruction_only) or (text == input_only):
                    return False, False, 1,1;
            
                    
            if "list" in rule.lower():
                sc = "Numbered-list" if structural_class == "List" else structural_class
                    
                ground_truth_distribution = list_distribution_modeling(full_output, list_type=sc) 
                dist = list_distribution_modeling(text, list_type=sc)
                structural_distance = structure_distance(dist, ground_truth_distribution)
                distance = (structural_distribution_weight * distance + input_ground_overlap_weight * input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output));

                if distance > max_distance_threshold or (text == instruction_only) or (text == input_only):
                    return False, False,distance,1;
                    

            elif "mathematical" in rule.lower():
                structural_distance = calculate_distance(text, full_output, structural_class);
                                
                distance = (structural_distribution_weight * structural_distance + input_ground_overlap_weight * input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output));

                if distance > max_distance_threshold or (text == instruction_only) or (text == input_only):
                    return False, False,distance,1
                
            elif "code" in rule.lower():
                structural_distance = calculate_distance(text, full_output, structural_class);

                distance = (structural_distribution_weight * structural_distance + input_ground_overlap_weight * input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output));

                if distance > max_distance_threshold or (text == instruction_only) or (text == input_only):
                    return False, False,distance,1

            
            elif "dialogue" in rule.lower():
                
                structural_distance = calculate_distance(text, full_output, structural_class);

                
                distance = (structural_distribution_weight * structural_distance + input_ground_overlap_weight * input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output));

                if distance > max_distance_threshold or (text == instruction_only) or (text == input_only):
                    return False, False, distance,1

            elif "question" in rule.lower():
                structural_distance = calculate_distance(text, full_output, structural_class);

                distance = (structural_distribution_weight * structural_distance + input_ground_overlap_weight * input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output));

                if distance > max_distance_threshold or (text == instruction_only) or (text == input_only):
                    return False, False,distance,1

            elif "number" in rule.lower():
                structural_distance = calculate_distance(text, full_output, structural_class);
                
                distance = (structural_distribution_weight * structural_distance + input_ground_overlap_weight * input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output));

                if distance > max_distance_threshold or (text == instruction_only) or (text == input_only):
                    return False, False,distance,1
                    
        return True, False,distance,1

    else:
        for rule in rule_based_info:
            
            if "table" in rule.lower():
                if has_table:
                    return False, False,0,0

            elif "list" in rule.lower():
                if not positive_sample:
                    return False, False,0,0
                    
                sc = "Numbered-list" if structural_class == "List" else structural_class
                    
                ref_dist      = list_distribution_modeling(full_output, list_type=sc)
                positive_dist = list_distribution_modeling(positive_sample, list_type=sc)
                neg_dist      = list_distribution_modeling(text, list_type=sc)

                pos_distance = structure_distance(ref_dist, positive_dist)
                neg_distance = structure_distance(ref_dist, neg_dist)

                pos_distance = (structural_distribution_weight*pos_distance + input_ground_overlap_weight*input_ground_overlap_distance(input_context, full_output, positive_sample) + output_variance_weight*output_variance_distance(full_output, positive_sample) + tokens_absolute_length_weight*tokens_absolute_length_difference(positive_sample, full_output))
                
                neg_distance = (structural_distribution_weight*neg_distance + input_ground_overlap_weight*input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output))

                margin = neg_distance - pos_distance
                
                if abs(margin) < max_distance:
                    return False, False,pos_distance, neg_distance 

                    
                return True, margin < -max_distance and (text != instruction_only) and (text != input_only), pos_distance, neg_distance


            elif structural_class == "Prose":
                if not positive_sample:
                    return False, False,1,1;
                    
                pos_distance = (structural_distribution_weight*pos_distance + input_ground_overlap_weight*input_ground_overlap_distance(input_context, full_output, positive_sample) + output_variance_weight*output_variance_distance(full_output, positive_sample) + tokens_absolute_length_weight*tokens_absolute_length_difference(positive_sample, full_output))
                
                neg_distance = (structural_distribution_weight*neg_distance + input_ground_overlap_weight*input_ground_overlap_distance(input_context, full_output, text) + output_variance_weight*output_variance_distance(full_output, text) + tokens_absolute_length_weight*tokens_absolute_length_difference(text, full_output))

                print("Prose neg distance", neg_distance)
                print("Prose pos distance", pos_distance)
                
                margin = neg_distance - pos_distance
                
                if abs(margin) < max_distance or (text == instruction_only) or (text == input_only):
                    return False, False,pos_distance, neg_distance 
                    
                return True, margin < -max_distance and (text != instruction_only) and (text != input_only), pos_distance, neg_distance

            
            elif "mathematical" in rule.lower():
                if not positive_sample:
                    return False, False,1,1

                margin, should_swap, pos_distance, neg_distance  = _compute_margin(
                    maths_distribution_modeling, full_output, positive_sample, text, input_context,structural_distribution_weight, tokens_absolute_length_weight, output_variance_weight, input_ground_overlap_weight, max_distance
                )
                
                if abs(margin) < max_distance:
                    return False, False,pos_distance,neg_distance;
                    
                return True, margin < -max_distance and (text != instruction_only) and (text != input_only), pos_distance, neg_distance

            
            elif "code" in rule.lower():
                if not positive_sample:
                    return False, False,0,0

                margin, should_swap,pos_distance, neg_distance = _compute_margin(
                    code_distribution_modeling, full_output, positive_sample, text, input_context, structural_distribution_weight, tokens_absolute_length_weight, output_variance_weight, input_ground_overlap_weight, max_distance
                )
                if abs(margin) < max_distance:
                    return False, False,pos_distance,neg_distance
                    
                return True,  margin < -max_distance and (text != instruction_only) and (text != input_only), pos_distance, neg_distance

                

            elif "dialogue" in rule.lower():
                if not positive_sample:
                    return False, False, 0,0

                margin, should_swap,pos_distance, neg_distance = _compute_margin(
                    dialogue_distribution_modeling, full_output, positive_sample, text, input_context, structural_distribution_weight, tokens_absolute_length_weight, output_variance_weight, input_ground_overlap_weight, max_distance
                )
                if abs(margin) < max_distance:
                    return False, False,pos_distance,neg_distance
                    
                return True, margin < -max_distance and (text != instruction_only) and (text != input_only), pos_distance, neg_distance

            
            elif "question" in rule.lower():
                if not positive_sample:
                    return False, False,0,0

                margin, should_swap,pos_distance, neg_distance = _compute_margin(
                    question_distribution_modeling, full_output, positive_sample, text, input_context,structural_distribution_weight, tokens_absolute_length_weight, output_variance_weight, input_ground_overlap_weight, max_distance
                )
                if abs(margin) < max_distance:
                    return False, False,pos_distance,neg_distance
                    
                return True, margin < -max_distance and (text != instruction_only) and (text != input_only), pos_distance, neg_distance
 
            elif "number" in rule.lower():
                if not positive_sample:
                    return False, False,0,0

                margin, should_swap,pos_distance, neg_distance = _compute_margin(
                    number_distribution_modeling, full_output, positive_sample, text, input_context,structural_distribution_weight, tokens_absolute_length_weight, output_variance_weight, input_ground_overlap_weight, max_distance
                )
                print(f"+++++++ Negative Number margin (threshold 0.1) ++++++++: {margin}")
                if abs(margin) < max_distance:
                    return False, False,pos_distance,neg_distance
                return True, margin < -max_distance and (text != instruction_only) and (text != input_only), pos_distance, neg_distance
                    
        return True, False,pos_distance,neg_distance
        


def create_prompts(args, instruction, input_context, output, structural_class,revealing_rate = 0, positive_sample=""):
    """
    Create positive and negative prompts based on structural class.
    """
    context = f"\n\n{input_context}" if input_context.strip() else ""
    
    rule_based_info = []
    
    if structural_class == "List":
        rule_based_info.append("include any list of multiple elements.")
    
    if structural_class == "Dashed-list":
        rule_based_info.append("include a dashed list of elements using -.")
    
    if structural_class == "Numbered-list":
        rule_based_info.append("include a numbered list of elements.");
        
    if structural_class == "Comma-list":
        rule_based_info.append("include a comma list of elements separated by commas (,).");
        
    if structural_class == "Table":
        rule_based_info.append("include a table structure.")
        
    elif structural_class == "Questions":
        rule_based_info.append("be a question and contain a question mark (?).")
        
    elif structural_class == "Mathematics":
        rule_based_info.append("include mathematical formulas, expressions or equations.")
    
    elif structural_class == "Code":
        rule_based_info.append("include code.")
        
    elif structural_class == "Number":
        rule_based_info.append("be a number only.")
        
    elif structural_class == "Dialog":
        rule_based_info.append("include a dialogue between two or more speakers.")
    
    elif structural_class == "Prose":
        rule_based_info.append("provide a correct answer that fully satisfies the instructions, written as plain prose only.")

    revealed_output = ""
    
    if output and output.strip() and revealing_rate > 0:
        output_words = output.split()
        output_length = len(output_words)
        reveal_length = int(output_length * revealing_rate)
        
        if reveal_length < 1:
            reveal_length = 1
            
        revealed_text = ' '.join(output_words[:reveal_length])
        revealed_output = f"{revealed_text}"

    if positive_sample:
        # STAGE 2: Creating NEGATIVE prompt
        if rule_based_info:
            if "list" in structural_class.lower():
                negative_prompt = negative_aligning_prompt.format(
                    Input=instruction, 
                    Context=context,
                    RuleBasedInfo=f""
                )
            elif structural_class == "Questions":
                negative_prompt = negative_aligning_prompt.format(
                    Input=instruction, 
                    Context=context,
                    RuleBasedInfo=""
                )
            else:
                negative_prompt = negative_aligning_prompt.format(
                    Input=instruction, 
                    Context=context,
                    RuleBasedInfo=""
                )
        else:
            negative_prompt = negative_aligning_prompt.format(
                Input=instruction, 
                Context=context,
                RuleBasedInfo=""
            )
        
        return ['rejected'], [negative_prompt], revealed_output, rule_based_info, structural_class
    
    else:
        # STAGE 1: Creating POSITIVE prompt
        if rule_based_info:
            if revealed_output:
                positive_prompt = positive_aligning_prompt.format(
                    Input=instruction, 
                    Context=context,
                    RuleBasedInfo=f"\n The response should {rule_based_info[0]}\n",
                    RevealedOutput=f"{revealed_output}"
                )
            else:
                positive_prompt = positive_aligning_prompt.format(
                    Input=instruction, 
                    Context=context,
                    RuleBasedInfo=f"\n The response should {rule_based_info[0]}\n",
                    RevealedOutput=""
                );
                
        else:
            if revealed_output:
                positive_prompt = positive_aligning_prompt.format(
                    Input=instruction, 
                    Context=context,
                    RuleBasedInfo="",
                    RevealedOutput=f"{revealed_output}"
                )
            else:
                positive_prompt = positive_aligning_prompt.format(
                    Input=instruction, 
                    Context=context,
                    RuleBasedInfo="",
                    RevealedOutput=""
                )
            
        return ['chosen'], [positive_prompt], revealed_output, rule_based_info, structural_class;




def ready_for_analysis(args, analysis_waiting_entry):
    return all([key in analysis_waiting_entry for key in ['chosen', 'rejected']])


def analyze_results(args, batch, model, tokenizer):
    results = []
    for entry in batch:

        if entry.data['revealed_output']:
            chosen_full = entry.data['revealed_output'] + " " + entry.data['chosen']
        else:
            chosen_full = entry.data['chosen']
        
        rejected_full = entry.data['rejected'];
        
        results.append({
            'prompt': entry.data.get('prompt', ''),
            'input': entry.data.get('input', ''),
            'actual_prompts': entry.data.get('actual_prompts', []),
            'chosen_result': chosen_full.strip(),
            'rejected_result': rejected_full.strip(),
            'rule_based_info': entry.data.get('rule_based_info', []),
            'structural_class': entry.data.get('structural_class', 'Unknown'),
            'contrastive_status': entry.data.get('contrastive_status', 'Contrastive'),
            'id': entry.data.get('id', None),
            'ground_truth': entry.data.get('ground_truth', ''),
            'positive_distance': entry.data.get('positive_distance', None),
            'negative_distance': entry.data.get('negative_distance', None),
            'status': True
        })
    return results


# ---------------------------------------------------------------------------
# JSONL helpers – replace per-index .json files with a single .jsonl file
# ---------------------------------------------------------------------------

def load_completed_indices(jsonl_path):
    """Return the set of integer indices already saved in *jsonl_path*.

    Each line is expected to be a JSON object that was written by
    ``append_result_to_jsonl``.  Lines that are missing an ``_idx`` field or
    that cannot be parsed are silently skipped so that a partially-written
    file never blocks a resume.
    """
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if '_idx' in obj:
                    completed.add(obj['_idx'])
            except json.JSONDecodeError:
                pass
    return completed


def append_result_to_jsonl(jsonl_path, idx, save_dict):
    """Append a single result dict (augmented with ``_idx``) to *jsonl_path*.

    The file is opened in append mode so concurrent / resumed runs never
    overwrite earlier results.
    """
    record = {'_idx': idx, **save_dict}
    with open(jsonl_path, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(record) + '\n')


# ---------------------------------------------------------------------------


class GenerationQueueEntry:
    def __init__(self, idx, key, retries, prompt, original_prompt, revealed_output, structural_class,
                 rule_based_info = None, require_validation=False, full_output="", input_context="",
                 positive_sample="", positive_sample_index=None, sample_id = 0):
        
        self.idx = idx
        self.key = key
        self.retries = retries
        self.prompt = prompt
        self.original_prompt = original_prompt
        self.revealed_output = revealed_output
        self.structural_class = structural_class
        self.rule_based_info = rule_based_info or []
        self.require_validation = require_validation
        self.full_output = full_output
        self.input_context = input_context
        self.positive_sample = positive_sample
        self.positive_sample_batch_index = positive_sample_index
        self.sample_id = sample_id



class AnalysisQueueEntry:
    def __init__(self, idx, data, revealed_output, rule_based_info, structural_class):
        self.idx = idx
        self.data = data
        self.data['revealed_output'] = revealed_output
        self.data['rule_based_info'] = rule_based_info
        self.data['structural_class'] = structural_class
        self.data['ground_truth'] = self.data.get('ground_truth', '')
        self.data['id'] = self.data.get('id', self.data.get('sample_id', None))
        self.data['positive_distance'] = self.data.get('positive_distance', None)
        self.data['negative_distance'] = self.data.get('negative_distance', None)





def main(
    response_len:int = 256,
    resampling_number:int = 5,
    use_basemodel: bool = False,
    checkpoint_dir: str = "",
    promptsFile: str = "data/sampled_alpaca_with_ids&prose.json",
    indices: str = None,
    indicesMod: int = 1,
    indicesRemainder: int = 0,
    batchSize: int = 32,
    maxRetries: int = 3,
    outDir: str = "",
    seed: int = 0,
    modelString: str = "meta-llama/Llama-3.2-3B",
    maxResamplingAttempts: int = 3,
    maxSimilarityPositive: float = 0.9,
    minSimilarityNegative: float = 0,
    structural_distribution_weight:float = 0.7,
    tokens_absolute_length_weight:float = 0.1,
    output_variance_weight:float = 0.1,
    input_ground_overlap_weight:float = 0.1,
    max_distance: dict = 0.25,
):


    

    args = Namespace(
        response_len = response_len,
        resampling_number = resampling_number,
        use_basemodel = use_basemodel,
        checkpoint_dir=checkpoint_dir,
        promptsFile=promptsFile,
        indices=indices,
        indicesMod=indicesMod,
        indicesRemainder=indicesRemainder,
        batchSize=batchSize,
        maxRetries=maxRetries,
        outDir=outDir,
        seed=seed,
        modelString=modelString,
        maxResamplingAttempts=maxResamplingAttempts,
        maxSimilarityPositive=maxSimilarityPositive,
        minSimilarityNegative=minSimilarityNegative,
        structural_cues_weights = structural_cues_weights,
    );


    


    ## Balancing data
    

    from collections import defaultdict
    import random
    import json
    
    ## Balancing data
    rng = random.Random(args.seed)
    
    with open(args.promptsFile, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Step 1: group by class
    class_groups = defaultdict(list)
    
    for item in data:
        cls = item.get('class', 'Prose')
        if cls.lower() not in ["table","list"]:
            class_groups[cls].append(item)
    
    # Safety check (avoid crash if empty)
    if not class_groups:
        raise ValueError("No classes found after filtering.")
    
    # Step 2: find largest class size (target)
    
    max_size = max(len(v) for v in class_groups.values());

    max_size = min(max_size, 10000);

    # Step 3: balance (upsample small, downsample large)
    balanced_items = []
    
    for cls, items in class_groups.items():
        if len(items) < max_size:
            # Upsample (with replacement)
            sampled = [rng.choice(items) for _ in range(max_size)]
        elif len(items) > max_size:
            # Downsample (without replacement)
            sampled = rng.sample(items, max_size)
        else:
            # Already equal
            sampled = items.copy()
        
        print(cls, len(items), "→", len(sampled))
        balanced_items.extend(sampled)

    # Step 4: expand into data_entries
    data_entries = []



    for item in balanced_items:
        for _ in range(1):
            data_entries.append({
                'instruction': item['instruction'].strip(),
                'input': item.get('input', '').strip(),
                'output': item.get('output', '').strip(),
                'class': item.get('class', 'Prose'),
                'id': item['id']
            })
            
    
    # Step 5: shuffle final dataset
    rng.shuffle(data_entries)
    print("++++++++Data entries length: ++++++++++",len(data_entries));


    os.makedirs(args.outDir, exist_ok=True)

    # ── Single JSONL output file (replaces per-index .json files) ────────────
    jsonl_path = os.path.join(args.outDir, 'generated_preference_data.jsonl');
    completed_indices = load_completed_indices(jsonl_path);
    print(f"[resume] {len(completed_indices)} indices already saved in {jsonl_path}");
    # ─────────────────────────────────────────────────────────────────────────

    
    tokenizer = AutoTokenizer.from_pretrained(args.modelString, padding_side='left')
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token


    
    if args.use_basemodel:
        
        model = AutoModelForCausalLM.from_pretrained(
            args.modelString, 
            device_map='auto', 
            quantization_config=bnb_config
        ).eval();

        
    else:
        
        model = load_4bit_model_for_inference(
            checkpoint_dir=args.checkpoint_dir,
            bits=4,
            fp16=False,
            bf16=True,
            gradient_checkpointing=False,
            adapter_name="lora_policy",
            is_trainable=False,
            reuse_base_model=True,
            trust_remote_code=True,
            base_model_mapping=None,
            fully_initialize=False,
            base_model_name_or_path_for_fully_initialize=args.modelString,
        )

    
    indices_list = parse_indices(args.indices) if args.indices is not None else list(range(len(data_entries)))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            
            current_idx = -1
            failed_indices = set()
            generation_queue = []
            analysis_waiting = defaultdict(lambda: {})
            analysis_queue = []
            resampling_attempts = defaultdict(lambda: {'chosen': 0, 'rejected': 0})

            while current_idx < len(data_entries) or len(generation_queue) > 0 or len(analysis_queue) > 0:
                # ----------------------------------------------------------
                # Run analysis batch
                # ----------------------------------------------------------
                if len(analysis_queue) >= args.batchSize or (len(analysis_queue) > 0 and current_idx >= len(data_entries)):
                    batch = analysis_queue[:args.batchSize]
                    analysis_queue = analysis_queue[args.batchSize:]
                    assert all([x.idx not in failed_indices for x in batch])
                    
                    save_dicts = analyze_results(args, batch, model, tokenizer)
                    
                    for entry, save_dict in zip(batch, save_dicts):
                        # ── CHANGED: append to JSONL instead of writing individual file ──
                        append_result_to_jsonl(jsonl_path, entry.idx, save_dict)
                        # ─────────────────────────────────────────────────────────────────
                        if entry.idx in resampling_attempts:
                            del resampling_attempts[entry.idx]

                
                # ----------------------------------------------------------
                # Run generation batch
                # ----------------------------------------------------------
                elif len(generation_queue) >= args.batchSize or (len(generation_queue) > 0 and current_idx >= len(data_entries)):
                    
                    batch = generation_queue[:args.batchSize]
                    generation_queue = generation_queue[args.batchSize:]
                    batch = [x for x in batch if x.idx not in failed_indices]
                    batch_prompts = [x.prompt for x in batch]
                    batch_results = generate_result(model, tokenizer, batch_prompts, max_new_tokens=args.response_len)

                    for i, (status, result) in enumerate(batch_results):
                     
                        if status:
                            is_positive = (batch[i].key == 'chosen')

                            is_valid, should_swap, positive_distance, negative_distance = validate_rule_based_info(
                                result, 
                                batch[i].rule_based_info,
                                batch[i].structural_class,
                                is_positive=is_positive,
                                revealed_output=batch[i].revealed_output,
                                full_output=batch[i].full_output,
                                positive_sample=batch[i].positive_sample,
                                max_distance_threshold=args.maxSimilarityPositive,
                                min_distance_threshold=args.minSimilarityNegative,
                                input_context=batch[i].original_prompt + " " + (batch[i].input_context if batch[i].input_context else ""),
                                input_only = batch[i].input_context if batch[i].input_context else "",
                                instruction_only = batch[i].original_prompt,
                                structural_distribution_weight = structural_distribution_weight,
                                tokens_absolute_length_weight = tokens_absolute_length_weight,
                                output_variance_weight = output_variance_weight,
                                input_ground_overlap_weight = input_ground_overlap_weight,                                
                                max_distance = max_distance,
                            )
                            
                            is_plausible = True;
                            perplexity = None;
                            
                            if is_positive:
                                
                                should_accept = is_valid;
                                
                                if not should_accept:
                                    
                                    resampling_attempts[batch[i].idx][batch[i].key] += 1;
                                    
                                    if resampling_attempts[batch[i].idx][batch[i].key] < args.maxResamplingAttempts:
                                        generation_queue.append(batch[i])
                                        continue;
                                        
                                    else:
                                        analysis_waiting[batch[i].idx]['contrastive_status'] = 'Non-contrastive'
                                        
                                        print(f"Warning: Max resampling attempts reached for idx {batch[i].idx} "
                                              f"(chosen/{batch[i].structural_class}) - couldn't satisfy structure "
                                              f"with distance ≤ {args.maxSimilarityPositive}. Marking as Non-contrastive.")
                                        
                                
                                analysis_waiting[batch[i].idx][batch[i].key] = result
                                analysis_waiting[batch[i].idx]['prompt'] = batch[i].original_prompt
                                analysis_waiting[batch[i].idx]['input'] = batch[i].input_context
                                analysis_waiting[batch[i].idx]['ground_truth'] = batch[i].full_output
                                analysis_waiting[batch[i].idx]['id'] = batch[i].sample_id
                                analysis_waiting[batch[i].idx]['positive_distance'] = positive_distance
                                analysis_waiting[batch[i].idx]['negative_distance'] = analysis_waiting[batch[i].idx].get('negative_distance', None)
                                
                                if 'actual_prompts' not in analysis_waiting[batch[i].idx]:
                                    analysis_waiting[batch[i].idx]['actual_prompts'] = []
                                    
                                    
                                analysis_waiting[batch[i].idx]['actual_prompts'].append(batch[i].prompt)
                                
                                if 'contrastive_status' not in analysis_waiting[batch[i].idx]:
                                    analysis_waiting[batch[i].idx]['contrastive_status'] = 'Contrastive'

                                instruction = batch[i].original_prompt
                                input_context = batch[i].input_context
                                structural_class = batch[i].structural_class
                                
                                if batch[i].revealed_output:
                                    positive_sample_full = batch[i].revealed_output + " " + result
                                else:
                                    positive_sample_full = result
                                
                                keys, new_prompts, revealed_output, rule_based_info, structural_class = create_prompts(
                                    args, instruction, input_context, batch[i].full_output, structural_class = structural_class, 
                                    positive_sample=positive_sample_full
                                );
                                
                                for key, new_prompt in zip(keys, new_prompts):
                                    
                                    generation_queue.append(GenerationQueueEntry(
                                        batch[i].idx, key, 0, new_prompt, instruction, revealed_output,
                                        structural_class=structural_class,
                                        rule_based_info=rule_based_info,
                                        require_validation=True,
                                        full_output=batch[i].full_output,
                                        input_context=input_context,
                                        positive_sample=positive_sample_full
                                    ))
                                    
                            else:
                                
                                should_accept = is_valid;
                                
                                if not should_accept:
                                    
                                    resampling_attempts[batch[i].idx][batch[i].key] += 1;
                                    
                                    if resampling_attempts[batch[i].idx][batch[i].key] < args.maxResamplingAttempts:
                                        generation_queue.append(batch[i]);
                                        continue
                                        
                                    else:
                                        print(f"Warning: Max resampling attempts reached for idx {batch[i].idx} "
                                              f"(rejected/{batch[i].structural_class}) - couldn't achieve distance ≥ "
                                              f"{args.minSimilarityNegative} to positive sample. Marking as Non-contrastive.")
                                        
                                        analysis_waiting[batch[i].idx]['contrastive_status'] = 'Non-contrastive'
                                
                                if should_swap and 'chosen' in analysis_waiting[batch[i].idx]:
                                    print(f"[idx {batch[i].idx}] Swapping chosen/rejected labels "
                                          f"(negative sample is more ref-like than positive sample).")

                                    analysis_waiting[batch[i].idx]['negative_distance'] = negative_distance
                                    analysis_waiting[batch[i].idx]['positive_distance'] = positive_distance
                                    analysis_waiting[batch[i].idx]['rejected'] = result

                                    print("before swapping",analysis_waiting[batch[i].idx]);
                                    
                                    old_chosen = analysis_waiting[batch[i].idx].pop('chosen')
                                    analysis_waiting[batch[i].idx]['chosen'] = result
                                    analysis_waiting[batch[i].idx]['rejected'] = old_chosen

                                    old_pos_dist = analysis_waiting[batch[i].idx].pop('positive_distance', None)
                                    old_neg_dist = analysis_waiting[batch[i].idx].pop('negative_distance', None)
                                    if old_pos_dist is not None:
                                        analysis_waiting[batch[i].idx]['negative_distance'] = positive_distance
                                    analysis_waiting[batch[i].idx]['positive_distance'] = negative_distance

                                    print("After swapping",analysis_waiting[batch[i].idx])
                                else:
                                    analysis_waiting[batch[i].idx]['rejected'] = result
                                    analysis_waiting[batch[i].idx]['negative_distance'] = negative_distance
                                    analysis_waiting[batch[i].idx]['positive_distance'] = positive_distance

                                if 'actual_prompts' not in analysis_waiting[batch[i].idx]:
                                    analysis_waiting[batch[i].idx]['actual_prompts'] = []
                                analysis_waiting[batch[i].idx]['actual_prompts'].append(batch[i].prompt)
                                
                                if perplexity is not None:
                                    if 'rejected_perplexity' not in analysis_waiting[batch[i].idx]:
                                        analysis_waiting[batch[i].idx]['rejected_perplexity'] = perplexity
                                
                                if 'contrastive_status' not in analysis_waiting[batch[i].idx]:
                                    analysis_waiting[batch[i].idx]['contrastive_status'] = 'Contrastive'
                                
                                if ready_for_analysis(args, analysis_waiting[batch[i].idx]):
                                    analysis_queue.append(AnalysisQueueEntry(
                                        batch[i].idx, 
                                        analysis_waiting[batch[i].idx], 
                                        batch[i].revealed_output, 
                                        batch[i].rule_based_info, 
                                        batch[i].structural_class,   
                                    ))
                                    del analysis_waiting[batch[i].idx]

                        else:
                            batch[i].retries += 1
                            if batch[i].retries >= args.maxRetries:
                                failed_indices.add(batch[i].idx)
                                # ── CHANGED: append failure record to JSONL ──────────────
                                append_result_to_jsonl(jsonl_path, batch[i].idx, {'status': False})
                                # ─────────────────────────────────────────────────────────
                                if batch[i].idx in analysis_waiting:
                                    del analysis_waiting[batch[i].idx]
                                if batch[i].idx in resampling_attempts:
                                    del resampling_attempts[batch[i].idx]
                            else:
                                generation_queue.append(batch[i])
                
                # ----------------------------------------------------------
                # Add next data entry to generation queue (POSITIVE ONLY)
                # ----------------------------------------------------------
                else:
                    
                    current_idx += 1;
                    
                    if current_idx >= len(data_entries) or current_idx not in indices_list or current_idx % args.indicesMod != args.indicesRemainder:
                        continue
                        
                    # ── CHANGED: skip via JSONL completed set instead of file existence ──
                    
                    if current_idx in completed_indices:
                        continue

                    instruction    = data_entries[current_idx]['instruction']
                    input_context  = data_entries[current_idx]['input']
                    output         = data_entries[current_idx]['output']
                    structural_class = data_entries[current_idx]['class']
                    sample_id = data_entries[current_idx]['id']
                    
                    keys, new_prompts, revealed_output, rule_based_info, structural_class = create_prompts(
                        args, instruction, input_context, output, structural_class = structural_class
                    )
                    
                    require_validation = True
                    
                    for key, new_prompt in zip(keys, new_prompts):
                        generation_queue.append(GenerationQueueEntry(
                            current_idx, key, 0, new_prompt, instruction, revealed_output,
                            structural_class=structural_class,
                            rule_based_info=rule_based_info,
                            require_validation=require_validation,
                            full_output=output,
                            input_context=input_context,
                            positive_sample="",
                            sample_id = sample_id
                        ))
                        


if __name__ == "__main__":
    fire.Fire(main)