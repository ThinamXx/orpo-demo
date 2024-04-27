# Reference:
# --> https://mlabonne.github.io/blog/posts/2024-04-19_Fine_tune_Llama_3_with_ORPO.html
# --> https://github.com/xfactlab/orpo/tree/main

import os
import time

import wandb
import torch
import argparse
import subprocess

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from peft import LoraConfig, get_peft_model
from trl import ORPOTrainer, setup_chat_format

from orpo.src.args import default_args
from orpo.src.utils import preprocess_logits_for_metrics


class ORPO(object):

    def __init__(self, args) -> None:
        self.start = time.gmtime()
        self.args = args

        # Load Tokenizer
        print(">>> 1. Loading Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name, cache_dir=self.args.cache_dir
        )
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
            print("     1-1. Chat Template Applied (<|user|> <|assistant|>)")
        else:
            pass

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load Model
        print(">>> 2. Loading Model")
        if self.args.enable_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        if self.args.flash_attention_2 and torch.cuda.get_device_capability()[0] >= 8:
            subprocess.run(["pip", "install", "-qqq", "flash-attn"])
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                cache_dir=self.args.cache_dir,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                cache_dir=self.args.cache_dir,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        if self.args.enable_lora:
            peft_config = LoraConfig(
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                r=self.args.lora_rank,
                task_type="CAUSAL_LM",
            )

            self.model.enable_input_require_grads()

            self.model = get_peft_model(self.model, peft_config=peft_config)

            print("     2-1. QLoRA & LoRA adapter applied!")
            self.model.print_trainable_parameters()
        else:
            pass

        # Prepare chat format.
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)

        # Load Dataset
        print(">>> 3. Loading Dataset")
        data = load_dataset(self.args.data_name, split="all")
        data = data.shuffle(seed=self.args.seed).select(
            range(1000)
        )  # selecting only 1000 samples for testing.

        # Preprocess Dataset
        print(">>> 4. Preprocessing Dataset")
        self.dataset = data.map(
            self.preprocess_dataset,
            batched=True,
            num_proc=self.args.num_proc,
        )

        self.dataset = self.dataset.train_test_split(
            test_size=0.01, seed=self.args.seed
        )
        self.is_test = True  # always set to True.
        self.train = self.dataset["train"]
        self.test = self.dataset["test"]

        # Set WANDB & Logging Configurations
        self.run_name = f"{self.args.model_name.split('/')[-1]}-{self.args.data_name.split('/')[-1]}-lambda{self.args.alpha}-ORPO-{self.start.tm_mday}-{self.start.tm_hour}-{self.start.tm_min}"
        self.save_dir = os.path.join(
            "./checkpoints/", f"{self.args.data_name.split('/')[-1]}/{self.run_name}"
        )
        self.log_dir = os.path.join(
            "./checkpoints/",
            f"{self.args.data_name.split('/')[-1]}/{self.run_name}/logs",
        )

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def preprocess_dataset(self, example):
        example["chosen"] = self.tokenizer.apply_chat_template(
            example["chosen"], tokenize=False
        )
        example["rejected"] = self.tokenizer.apply_chat_template(
            example["rejected"], tokenize=False
        )
        return example

    def prepare_trainer(self):
        wandb.init(name=self.run_name)
        arguments = TrainingArguments(
            output_dir=self.save_dir,  # output directory
            logging_dir=self.log_dir,
            logging_steps=50,
            learning_rate=self.args.lr,
            overwrite_output_dir=True,  # overwrite the content of the output directory
            num_train_epochs=self.args.num_train_epochs,  # number of training epochs
            per_device_train_batch_size=self.args.per_device_train_batch_size,  # batch size for training
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,  # batch size for evaluation
            evaluation_strategy=self.args.evaluation_strategy if self.is_test else "no",
            save_strategy=self.args.evaluation_strategy,
            optim=self.args.optim,
            warmup_steps=self.args.warmup_steps,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": False if self.args.enable_lora else True
            },
            load_best_model_at_end=self.is_test,
            do_train=True,
            do_eval=self.is_test,
            lr_scheduler_type=self.args.lr_scheduler_type,
            remove_unused_columns=False,
            report_to="wandb",
            run_name=self.run_name,
            bf16=True,
            seed=self.args.seed,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        self.trainer = ORPOTrainer(
            model=self.model,
            alpha=self.args.alpha,
            pad=self.tokenizer.pad_token_id,
            disable_prompt_loss=self.args.disable_prompt_loss,
            args=arguments,
            train_dataset=self.train,
            eval_dataset=self.test if self.is_test else None,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def run(self):
        print(">>> 5. Preparing ORPOTrainer")
        self.prepare_trainer()

        if self.args.disable_prompt_loss:
            print("Discarding Prompt Tokens for NLL Loss")
        self.trainer.train()

        # Saving code for FSDP
        if self.trainer.is_fsdp_enabled:
            self.trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
                "FULL_STATE_DICT"
            )
        self.trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ORPO")
    args = default_args(parser)

    # Set the random seed for the entire pipeline
    set_seed(args.seed)

    # Set WANDB configurations
    if args.wandb_entity is not None and args.wandb_project_name is not None:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
        os.environ["WANDB_PROJECT"] = args.wandb_project_name
    else:
        pass
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(
        "================================================================================================\n"
    )
    print(f">>> Fine-tuning {args.model_name} with ORPO on {args.data_name}\n")
    print(
        "================================================================================================"
    )
    print("\n\n>>> Summary:")
    print(f"    - Lambda              : {args.alpha}")
    print(f"    - Training Epochs     : {args.num_train_epochs}")
    print(f"    - Prompt Max Length   : {args.prompt_max_length}")
    print(f"    - Response Max Length : {args.response_max_length}")

    item = ORPO(args=args)
    item.run()
