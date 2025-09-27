#!/usr/bin/env python3
"""
Multi-GPU Language Model Fine-tuning with Unsloth
Author: AI Engineering Team
Description: Distributed training script for cost-effective LLM fine-tuning
"""

import os
import torch
import warnings
import subprocess
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import yaml
import argparse

from huggingface_hub import HfFolder, login
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only
from unsloth import is_bf16_supported

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
logging.basicConfig(level=logging.INFO)

class MultiGPUTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_environment()
        self.setup_distributed_training()
        
    def setup_environment(self):
        """Configure environment variables and HuggingFace authentication"""
        # HuggingFace setup
        if HfFolder.get_token() is None:
            login()
            
        # Set cache directories
        os.environ["HF_HOME"] = "/workspace"
        os.environ["HF_HUB_CACHE"] = "/workspace/hub"
        
        # Unsloth optimizations
        os.environ['UNSLOTH_COMPILE_DISABLE'] = '1'
        os.environ["UNSLOTH_DISABLE_CACHE"] = '1'
        os.environ['UNSLOTH_DISABLE_RL_PATCH'] = "1"
        
        print(f"HF_HOME: {os.environ['HF_HOME']}")
        
    def setup_distributed_training(self):
        """Initialize distributed training variables"""
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))
        self.is_main_process = self.local_rank in (-1, 0)
        
        if self.is_main_process:
            print(f"Running on {self.world_size} GPUs.")
            
    def setup_tensorboard(self):
        """Launch TensorBoard for monitoring"""
        if not self.is_main_process:
            return
            
        logdir = "./logs"
        port = "6006"
        
        try:
            tb_process = subprocess.Popen(
                ["tensorboard", f"--logdir={logdir}", "--port", port, "--bind_all"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ,
            )
            print(f"TensorBoard running on port {port} (PID {tb_process.pid})")
            print(f"Access via: https://<pod-id>:{port}/proxy/{port}/")
        except FileNotFoundError:
            print("TensorBoard not found. Install with: pip install tensorboard")
            
    def load_model_and_tokenizer(self):
        """Load and configure the model with PEFT"""
        model_config = self.config['model']
        
        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config['name'],
            max_seq_length=model_config['max_seq_length'],
            load_in_4bit=model_config.get('load_in_4bit', False),
            load_in_8bit=model_config.get('load_in_8bit', False),
            torch_dtype=model_config.get('dtype'),
            device_map={torch.cuda.current_device()},
            use_gradient_checkpointing="unsloth",
        )
        
        if self.is_main_process:
            memory_gb = torch.cuda.memory_allocated() / 1e9
            print(f"Model loaded on device {torch.cuda.current_device()} with {memory_gb:.2f} GB")
            print(f"Tokenizer padding side: {tokenizer.padding_side}")
            
        # Configure PEFT (LoRA)
        lora_config = self.config['lora']
        model = FastLanguageModel.get_peft_model(
            model,
            rank=lora_config['rank'],
            lora_alpha=lora_config['alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config.get('dropout', 0),
            bias="none",
            random_state=3074,
            use_rslora=True,
        )
        
        if self.is_main_process:
            trainable_params = model.get_trainable_params() / 1e6
            print(f"PEFT LoRA model with {trainable_params:.2f}M trainable parameters")
            
        return model, tokenizer
        
    def load_dataset(self):
        """Load and prepare training/evaluation datasets"""
        dataset_config = self.config['dataset']
        
        # Load training dataset
        ft_data = load_dataset(dataset_config['name'])
        ft_train_data = ft_data["train"]
        
        # Load evaluation dataset
        if dataset_config.get('eval_dataset'):
            eval_data = load_dataset(dataset_config['eval_dataset'])
            eval_data = eval_data.get("test", eval_data.get("validation", eval_data["train"]))
        else:
            eval_data = ft_data.get("test", ft_data.get("validation", ft_data["train"]))
            
        if self.is_main_process:
            print(f"Training samples: {len(ft_train_data)}")
            print(f"Evaluation samples: {len(eval_data)}")
            
        return ft_train_data, eval_data
        
    def create_formatting_function(self, tokenizer):
        """Create function to format data for training"""
        def formatting_func(batch):
            """Convert batch of rows to list of chat-formatted prompts"""
            out = []
            for user_raw, assistant_raw in zip(batch['question'], batch['answer']):
                messages = [
                    {"role": "user", "content": user_raw},
                    {"role": "assistant", "content": assistant_raw},
                ]
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=self.config.get('enable_thinking', False),
                )
                
                # Remove BOS token if present
                bos = tokenizer.bos_token or "<bos>"
                if text.startswith(bos):
                    text = text[len(bos):]
                out.append(text)
            return out
            
        return formatting_func
        
    def setup_training_arguments(self):
        """Configure training arguments"""
        training_config = self.config['training']
        current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create run name
        model_name = self.config['model']['name'].split('/')[-1]
        dataset_name = self.config['dataset']['name']
        
        run_name = (f"{model_name}-ft-{dataset_name}-"
                   f"{training_config['epochs']}ep-"
                   f"{self.config['lora']['rank']}r-"
                   f"{training_config['batch_size']}b-"
                   f"{training_config['learning_rate']}lr-"
                   f"{current_timestamp}")
        
        if self.world_size > 1:
            run_name += f"-{self.world_size}GPU"
            
        if self.is_main_process:
            print(f"Run name: {run_name}")
            
        # Calculate gradient accumulation steps
        gradient_accumulation_steps = max(1, 
            training_config.get('target_batch_size', 32) // 
            training_config['batch_size'] // self.world_size
        )
        
        training_args = TrainingArguments(
            per_device_train_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=training_config['epochs'],
            learning_rate=training_config['learning_rate'],
            
            # Logging and evaluation
            logging_strategy="steps",
            logging_steps=max(1, int(0.05 * training_config['epochs'] * 100)),
            eval_strategy="steps" if 'eval_dataset' in self.config['dataset'] else "no",
            eval_steps=max(1, int(0.2 * training_config['epochs'] * 100)),
            per_device_eval_batch_size=training_config['batch_size'],
            
            # Mixed precision
            bf16=is_bf16_supported() if self.config['model'].get('dtype') is None else False,
            fp16=not is_bf16_supported() if self.config['model'].get('dtype') is None else False,
            
            # Output and reporting
            output_dir="outputs",
            run_name=run_name,
            logging_dir=f"./logs/{run_name}",
            report_to=["tensorboard"] if self.is_main_process else None,
            
            # Memory optimization
            remove_unused_columns=True,
            lr_scheduler_type="cosine",
            
            # DDP arguments
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
        )
        
        return training_args, run_name
        
    def get_chat_template(self, model_name: str):
        """Get appropriate chat template for model"""
        templates = {
            "llama": {
                "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
                "response": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            },
            "gemma": {
                "instruction": "<start_of_turn>user\n",
                "response": "<start_of_turn>model\n",
            },
            "qwen": {
                "instruction": "<|im_start|>user\n",
                "response": "<|im_start|>assistant\n<think>\n\n</think>\n",
            }
        }
        
        model_name_lower = model_name.lower()
        for key, template in templates.items():
            if key in model_name_lower:
                logging.info(f"Using {key} chat template")
                return template["instruction"], template["response"]
                
        logging.info("Using default chat template")
        return "", ""
        
    def print_memory_stats(self, stage: str):
        """Print GPU memory statistics"""
        if not self.is_main_process:
            return
            
        gpu_props = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
        allocated = round(torch.cuda.memory_allocated() / 1e9, 3)
        reserved = round(torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 3)
        
        print(f"\n--- {stage} Memory Stats ---")
        print(f"GPU: {gpu_props.name}")
        print(f"Total Memory: {max_memory} GB")
        print(f"Allocated: {allocated} GB")
        print(f"Reserved: {reserved} GB")
        print(f"Utilization: {(allocated/max_memory)*100:.1f}%")
        print("-" * 30)
        
    def train(self):
        """Main training loop"""
        self.print_memory_stats("Start")
        self.setup_tensorboard()
        
        # Load model and data
        model, tokenizer = self.load_model_and_tokenizer()
        train_dataset, eval_dataset = self.load_dataset()
        formatting_func = self.create_formatting_function(tokenizer)
        
        # Setup training
        training_args, run_name = self.setup_training_arguments()
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if self.config['dataset'].get('eval_dataset') else None,
            formatting_func=formatting_func,
        )
        
        # Configure response-only training
        instruction_tag, response_tag = self.get_chat_template(self.config['model']['name'])
        if instruction_tag and response_tag:
            trainer = train_on_responses_only(
                trainer,
                instruction_part=instruction_tag,
                response_part=response_tag,
            )
            
        self.print_memory_stats("Pre-Training")
        
        if self.is_main_process:
            print("Starting trainer.train()")
            
        # Start training
        trainer_stats = trainer.train()
        
        self.print_memory_stats("Post-Training")
        
        # Print training statistics
        if self.is_main_process:
            runtime = trainer_stats.log_history[-1].get("train_runtime", 0)
            print(f"\nTraining completed in {runtime:.2f} seconds")
            
            # Save model
            if self.config.get('save_model', False):
                save_path = f"./outputs/{run_name}"
                model.save_pretrained_merged(save_path, tokenizer, save_method="merged_16bit")
                print(f"Model saved to {save_path}")
                
        return trainer_stats

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Multi-GPU LLM Fine-tuning with Unsloth")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--test-run", action="store_true", help="Run with small dataset for testing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override for test run
    if args.test_run:
        config['dataset']['max_samples'] = 32
        config['training']['epochs'] = 1
        print("Running in test mode with limited data")
    
    # Initialize and run training
    trainer = MultiGPUTrainer(config)
    trainer_stats = trainer.train()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
