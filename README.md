# Low-Resource LLM Alignment

Code base for **"LLMs Alignment in Low Resource Settings"** - A comprehensive framework for aligning large language models using reinforcement learning and preference learning techniques optimized for resource-constrained environments.

## Overview

This repository implements an end-to-end pipeline for aligning large language models (LLMs) to human preferences through a structural curriculum learning approach. The system is specifically optimized for low-resource settings, using **QLoRA (Quantized Low-Rank Adaptation)**.

## Key Features

### 1. **Curriculum-Based Training Loop**
- Iterative refinement of policy through multiple rounds
- Dynamic weighting of structural cues (distribution, length, overlap, variance)
- Automatic checkpoint selection based on training metrics
- State tracking for resumable training

### 2. **Synthetic Preference Data Generation**
- Rule-based generation guided by structural constraints
- Support for multiple structural types (lists, tables, code, math, dialogue, prose)
- Contrastive learning with margin-based validation
- Demonstration-based augmentation for improved quality (In latest iterations)

### 3. **Reward Model Training**
- Binary preference modeling on synthetic data
- QLoRA-based efficient fine-tuning
- Transfer learning from previous checkpoints

### 4. **RL Policy Training (PPO)**
- Proximal Policy Optimization.
- reward model support
- Reference policy management
- Length bonus and token-level penalization
- RLVR support (Reinforcement Learning with Value Rewards) (still under testing)
- Ensemble reward modeling support (For experimentation with multiple reward models)

### 5. **Evaluation**
- Multiple evaluation datasets: Alpaca, HumanEval, IFEval, TruthfulQA
- Best-of-N sampling for improved performance
- Both greedy and sampling-based decoding

<!-- ## Repository Structure

```
├── src/                                    # All training scripts (organized)
│   ├── Curriculum_checkpoint_selection.py  # Checkpoint scoring and selection
│   ├── DPOtraining.py                      # Direct Preference Optimization
│   ├── Experiments_run.py                  # Main orchestrator (at root)
│   ├── Ground_truth_based_optimized_synthetic_generation.py  # Synthetic data gen
│   ├── PPO_training_base.py                # Proximal Policy Optimization
│   ├── RL_data_creation_PPO.py             # PPO training data preparation
│   ├── reward_model_base_training.py       # Reward model training
│   ├── supervised_fine_tuning.py           # Supervised fine-tuning baseline
│   ├── Synthetic_post_process.py           # Post-processing synthetic data
│   └── weighted_rl_scheduling.py           # Weighted sampling utilities
├── models/                                 # Training model implementations
│   ├── instruction_tuned_model.py          # Instruction-following model (user mainly for SFT)
│   ├── ppo_trainer_base.py                 # PPO trainer (Built on AlpacaFarm and SALMON)
│   ├── qlora_model.py                      # QLoRA quantization utilities
│   ├── reward_model.py                     # Reward model
│   ├── rl_models.py                        # RL-specific model utilities
│   ├── rl_trainer.py                       # Core RL training logic
│   └── trainer_utils.py                    # Training utilities
├── data_utils/                             # Data loading and processing
│   ├── common_utils.py                     # Shared utilities
│   ├── data_utils_ppo.py                   # PPO data loading
│   ├── data_utils_rm.py                    # Reward model data loading
│   ├── instruction_fine_tuning.py          # SFT data preparation
│   ├── reward_preference_data.py           # Preference data utilities
│   ├── RL_data_creation.py                 # RL training data creation
│   └── Synthetic_post_process.py           # Synthetic data post-processing
├── Evaluations/                            # Evaluation frameworks
│   ├── Checkpoint_generations_for_evaluation.py
│   ├── TruthfulQAGeneration_MC1.py
│   └── Eval_data/                          # Evaluation datasets
├── rule_based_contrastive_sampling/        # Structural constraint utilities
│   └── utils.py
├── prompts/                                # System and instruction prompts
│   ├── policy_meta_prompt_pattern.txt
│   └── sft_reward_prompt-2.txt
├── scripts/                                # Bash runner scripts
│   ├── ppo_training_run.sh
│   ├── reward_model_training_script_run.sh
│   ├── run_evaluation_generations.sh
│   ├── run_post_processing.sh
│   ├── synthetic_preference_data_generation.sh
│   └── TruthfulQA_evaluation.sh
├── data/                                   # Input and generated data
│   ├── alpaca_data.json                    # Base instruction dataset
│   ├── PPO_balanced_scheduled_training_data.json
│   └── sampled_alpaca_with_ids&prose.json
├── Experiments_run.py                      # Main entry point (ROOT LEVEL)
├── batch_job_script.sh                     # HPC job submission
├── curriculum_state.json                   # Resumable training state
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
``` -->

## Installation

### Prerequisites
- Python 3.10+
- 16GB VRAM recommended (tested on a Tesla V100)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Low_resource_llm_alignment
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face token:**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```


## Quick Start

### Running the Full Pipeline

```bash
python Experiments_run.py
```

This will:
1. Generate synthetic preference data based on the current policy, current structural level.
2. Post-process and balance the synthetic data
3. Train a reward model
4. Run PPO training
5. Evaluate on multiple benchmarks
6. Select the best checkpoint for the next iteration

### Single Component Execution

#### 1. Generate Synthetic Preference Data
```bash
bash scripts/synthetic_preference_data_generation.sh \
    --outDir data/iteration_0 \
    --promptsFile data/sampled_alpaca_with_ids&prose.json \
    --checkpoint_dir checkpoints/policy_v1 \
    --maxDistancePositive 0.6
```

#### 2. Train Reward Model
```bash
bash scripts/reward_model_training_script_run.sh \
    --dataset_path data/preference_data.csv \
    --output_dir reward_models/rm_v1 \
    --per_device_train_batch_size 8
```

#### 3. PPO Policy Training
```bash
bash scripts/ppo_training_run.sh \
    --model_directory meta-llama/Llama-3.2-3b \
    --reward_model_name_or_path reward_models/rm_v1 \
    --dataset_path data/rl_training_data.json \
    --output_dir PPO/policy_v1
```

#### 4. Evaluation
```bash
bash scripts/run_evaluation_generations.sh \
    --dataset_type alpaca \
    --checkpoint_dir PPO/policy_v1/adapter_model/lora_policy \
    --outFile evaluations/results.jsonl
```



## Data Formats

### Input Dataset (Structural Alpaca IF dataset)
```json
{
  "id": "unique_id",
  "instruction": "What is machine learning?",
  "input": "optional_context",
  "output": "ground_truth_answer",
  "structural_class": "Prose|Numbered-list|Table|..."
}
```

### Synthetic Preference Data
```csv
instruction,input,chosen,rejected,preference,contrastive_status,structural_class,positive_distance,negative_distance,ground_truth
```



## Evaluation Metrics

### Supported Benchmarks
- **Alpaca**: Instruction-following and helpfulness quality (gitHub: [Bench](https://github.com/tatsu-lab/alpaca_eval))
- **HumanEval**: Code generation capability (gitHub: [Bench](https://github.com/openai/human-eval))
- **IFEval**: Instruction following (gitHub: [Bench]([text](https://github.com/google-research/google-research/tree/master/instruction_following_eval)))
- **TruthfulQA**: Factual accuracy.


## Troubleshooting

### CUDA Out of Memory
- Reduce batch sizes
- Enable gradient checkpointing
- Use mixed precision training (`--fp16 True`)

### Proximal Policy Optimization Instability Fixes
- Reduce learning rate (try 1e-6)
- Lower KL coefficient (`--kl_coef 0.01`)


## Citation

If you use this code for your research, please cite our paper:


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements
In our work, we relied on adapting several open-source codebases to our setup, including:
- RLCD as a building block for preference data generation (link: [RLCD](https://github.com/facebookresearch/RLCD))
- AlpacaFarm and SALMON for Proximal Policy Optimization. (link: [AlpacaFarm](https://github.com/tatsu-lab/alpaca_farm), [SALMON](https://github.com/IBM/SALMON))