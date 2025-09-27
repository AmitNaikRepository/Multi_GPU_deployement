
from huggingface_hub import HfFolder, login
# Check if a token is already saved
if HfFolder.get_token() is None:
    login() #Will prompt only if not logged in
import os
os.environ ["HF_HOME"] = "/workspace"
os.environ ["HF_HUB_CACHE"] = "/workspace/hub" # (recommended) override just the repo cache
print(os.environ ["HF_HOME"])

WORLD_SIZE=int(os.environ.get('WORLD_SIZE', 1))
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1))
IS_MAIN_PROCESS = LOCAL_RANK in (-1, 0)

if IS_MAIN_PROCESS:
    print(f"Running on {WORLD_SIZE} GPUs.")

import torch 
import os 
os.environ['UNSLOTH_COMPILE_DISABLE']='1'
os.environ["UNSLOTH_DISABLE_CACHE"]='1'
os.environ['UNSLOTH_DISABLE_RL_PATCH']="1"


if IS_MAIN_PROCESS:
    import subprocess
    logdir="./logs"
    port="6006"
    #launch tensorboard for logging in the background
    tb_process = subprocess.Popen(
        ["tensorboard", f"--logdir={logdir}", "--port", port, "--bind_all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ,
    )
    print(f"Tensorboard running on port {port} (PID {tb_process.pid}), open it via https://<pod-id>:{port}/proxy/{port}/ this is  requied port forwarding in your kubernetes setup.")


model_slug="Qwen/Qwen3-4B"


enable_thinking=False
test_run=False

max_seq_len=4096
dtype=None #torch.bfloat16
load_in_4bit=False #use the 4 bit quantization  
load_in_8bit=False


#training hyperparameters
rank=32
lora_alpha=32
per_device_train_batch_size=1
gradient_accumulation_steps=int(32/per_device_train_batch_size/WORLD_SIZE)
epochs=3
learning_rate=2e-4


#dataset
#to use the synthetic dataset for testing
ft_dataset="synthetic" #or "alpaca" or "sharegpt" or "custom"
synthetic_dataset_size=1024 #number of samples in the synthetic dataset
#us the alpaca dataset 



# #question /evalutation criteria for the synthetic dataset
# q_column="question"
# c_column="answer"
# a_column="answer" #evaluation criteria can use if the evaluation criteri has none 

different_eval_Dataset=None

#lets use the cuda without restarting the kernel
from unsloth import FastLanguageModel
import gc, inspect,sys
import warnings
warnings.filterwarnings("ignore",message="TypedStorage is deprecated")

model,tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_slug,
    max_seq_length=max_seq_len,
    load_in_4bit=load_in_4bit,
    load_in_8bit=load_in_8bit,
    torch_dtype=dtype,
    device_map={torch.cuda.current_device()},
    use_gradient_checkpointing="unsloth" ,)


if IS_MAIN_PROCESS:
    print(f"Model loaded on device {torch.cuda.current_device()} with {torch.cuda.memory_allocated()/1e9} GB")
    print(tokenizer.padding_side)
    print(model)

model=FastLanguageModel.get_peft_model(
    model,
    rank=rank,
    lora_alpha=lora_alpha,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0,#you can use any but it is optimize
    bias="none",
    random_state=3074,
    use_rslora=True,)

if IS_MAIN_PROCESS:
    print(f"PEFT LoRA model with {model.get_trainable_params()/1e6}M trainable parameters")
    print(model)    

from datasets import load_dataset
ft_data=load_dataset(ft_dataset)
ft_train_data=ft_data["train"]

if different_eval_Dataset is not None:
    eval_data=load_dataset(different_eval_Dataset)
    eval_data=eval_data["test"] if "test" in eval_data else eval_data["validation"] if "validation" in eval_data else eval_data["train"]
else:
    eval_data=ft_data["test"] if "test" in ft_data else ft_data["validation"] if "validation" in ft_data else ft_data["train"]

print(ft_train_data)
print(eval_data['question'][0]) 

def formatting_func(batch):
    "convert a batches of rows to list [str of chat formatted prompts ]"
    out=[]
    for user_raw,assistant_raw in zip(batch['question'],batch['answer']):
        messages=[{"role":"user","content":user_raw},
                    {"role":"assistant","content":assistant_raw},]
        
        text=tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,)
        
        bos=tokenizer.bos_token or "<bos>"
        if text.startswith(bos):
            text=text[len(bos):]
        out.append(text)
    return out

print(formatting_func(eval_data[:2]))

from trl import SFTTrainer, sftconfig
from transformers import DataCollatorForSeq2Seq,TrainingArguments
from transformers import get_scheduler
from datetime import datetime 

current_timestamp=datetime.now().strftime("%Y%m%d-%H%M%S")


if "ft_dataset_name" in globals() or "ft_dataset_name" in locals():
    if ft_dataset_name is not None:
        run_name=f"{model_slug.split('/')[-1]}-ft-{ft_dataset_name.split('/')[-1][:15]} - {epochs}ep-{rank}r-{per_device_train_batch_size}b-{learning_rate}lr-{current_timestamp}"
    else:
        run_name=f"{model_slug.split('/')[-1]}-ft-{ft_dataset}-{epochs}ep-{rank}r-{per_device_train_batch_size}b-{learning_rate}lr-{current_timestamp}"
else:
    run_name=f"{model_slug.split('/')[-1]}-ft-{ft_dataset}-{epochs}ep-{rank}r-{per_device_train_batch_size}b-{learning_rate}lr-{current_timestamp}"

print(f"Run name: {run_name}")


#decide if we evaluate during training or after training

do_eval=eval_data is not None #true when we have the data 

if IS_MAIN_PROCESS:
    if do_eval:
        print(f"Evaluation will be done during training on {len(eval_data)} samples")
    else:
        print("Evaluation will be done after training")

#build the training arguments 
from unsloth import is_bf16_supported

training_args=TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=epochs,
    logging_strategy="steps",
    logging_steps=0.05,
    eval_steps=0.2,
    per_device_eval_batch_size=per_device_train_batch_size,
    bf16=is_bf16_supported() if dtype is None else False,
    fp16=not is_bf16_supported() if dtype is None else False,
    report_to=["tensorboard"] if IS_MAIN_PROCESS else None,
    output_dir="outputs",
    remove_unused_columns=True,
    lr_scheduler_type="cosine",

    #DDP  args 
    ddp_find_unused_parameters=False ,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant":True}, #False is faster and uses less memory but may have bugs with some models
)


gpu_tag=f"{WORLD_SIZE}GPU" if WORLD_SIZE>1 else "cpu"
run_name=f"{run_name}-{gpu_tag}"
print(f"Final run name: {run_name}")


training_args.run_name=run_name
training_args.logging_dir=f"{logdir}/{run_name}"

trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ft_train_data if not test_run else ft_train_data.select(range(min(synthetic_dataset_size,32))),
    eval_dataset=eval_data if (do_eval and not test_run) else eval_data.select(range(min(len(eval_data), 32))),
    formatting_func=formatting_func,

)


if IS_MAIN_PROCESS:
    print(trainer.train_dataset)

from unsloth.chat_templates import train_on_responses_only

import logging
logging.basicConfig(level=logging.INFO)

templates={
    "llama":(
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "gemma":(
        "<start_of_turn>User:\n",
        "<start_of_turn>model:\n",
    ),
    "qwen":(
        "<|im_start|>user\n",
        "<|im_start|>assistant\n<think>\n\n</think>\n",
    )
}

def get_template(model_slug:str):
    model_slug_lower=model_slug.lower()
    for key , template in templates.items():
        if key in model_slug_lower:
            logging.info(f"Using {key} chat template")
            return template
        
    logging.info("Using default chat template")
    return None

result =get_template(model_slug)

if result:
    instuction_tag, response_tag = result
else:   
#fallback or raise error
    instuction_tag, response_tag = "", ""

trainer=train_on_responses_only(
    trainer,
    instruction_tag=instuction_tag,
    response_part=response_tag,)


if IS_MAIN_PROCESS:
    tokenizer.decode(trainer.train_dataset[0]['input_ids'])
    tokenizer.decode([tokenizer.pad_token_id if x==-100 else x for x in trainer.train_dataset[0]['labels']]).replace(tokenizer.pad_token," ")

#check memeor
#title show you how current memory states 
gpu_states=torch.cuda.get_device_properties(0)
start_gpu_memory=round(torch.cuda.memory_reserved()/1024/1024/1024,3)
max_memory=round(gpu_states.total_memory/1024/1024/1024,3)
print(f"gpu={gpu_states.name}, total memory={max_memory}GB")
print(f"Memory allocated at start {torch.cuda.memory_allocated()/1e9} GB")



#checck process group status 
import torch.distributed as dist
if IS_MAIN_PROCESS:
    print("dist available", dist.is_available(),"initlized", dist.is_initialized())

if dist.is_initialized():
    if IS_MAIN_PROCESS:
        print("backend:", dist.get_backend(),"world:", dist.get_world_size(),"rank:", dist.get_rank())

if IS_MAIN_PROCESS:
    print("starting trainer.train()")

trainer_state=trainer.train()


#final memroy result 
#@title show the final memory and the time stats 
used_memory=round(torch.cuda.memory_reserved()/1024/1024/1024/1024,3)
used_memory_for_lora=round(used_memory-start_gpu_memory,3)
used_percentage=round(used_memory_for_lora/max_memory*100,3)
lora_percentage=round(used_memory_for_lora/max_memory*100,3)

print(f"{trainer_state.log_history[-1]['train_runtime']} seconds used for the training")
print(f"peak reserved memory= {used_memory} GB")
print(f"peak reserve memory for training ={used_memory_for_lora}")
print(f"peak reserved memory % of max memory={used_percentage}")
print(f"peak reserved memory for training % of max memory={lora_percentage}")

if IS_MAIN_PROCESS:
    #manually shorten the run name and jsut re run the push if needed
    run_name=f"ddp_demo -{WORLD_SIZE}"

    #merge to 16 bit  recomended shouuld merge the dequintized model for the best accuracy
    org="amit naik"
    print("saving and pushing as {run_name} and {org}/{run_name} to the hub")

    #just save locally 
    if False: model.save_pretrained_merge(f"{run_name}",tokenizer,save_method="merged_16bit")

    #dave locally and push to hub 
    if False: model.push_to_hub_merge(f"{org}/{run_name}",tokenizer,save_method="merged_16bit")

    print(run_name)


if __name__=="__main__":
    print("trainig complete successfully")