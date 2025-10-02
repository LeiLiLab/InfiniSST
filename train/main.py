# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import set_seed

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler

from model.model import (
    SLlamaLightning, 
    SQwen25Lightning, 
    Qwen2ACLightning, 
    SeamlessLightning
)

MODEL_CLASSES = {
    "w2v2_llama31": SLlamaLightning,
    "w2v2_qwen25": SQwen25Lightning,
    "qwen2ac": Qwen2ACLightning,
    "seamless_llama31": SeamlessLightning
}

@dataclass
class SpeechEncoderArguments:
    # w2v2-llama
    w2v2_path: Optional[str] = field(default=None)
    w2v2_type: Optional[str] = field(default=None)
    w2v2_freeze: bool = field(default=False)
    ctc_finetuned: bool = field(default=False)
    length_shrink_cfg: str = field(default=None)
    xpos: bool = field(default=False)
    rope: bool = field(default=True)

    # qwen2-audio-chat
    whisper_freeze: bool = field(default=False)
    adapter_freeze: bool = field(default=False)

    # seamless-m4t-v2 encoder
    seamless_path: Optional[str] = field(default=None)
    seamless_freeze: bool = field(default=False)

    # common
    block_size: int = field(default=48)
    max_cache_size: int = field(default=500)

@dataclass
class ModelArguments:
    model_type: str = field(default="w2v2_llama31")
    llm_path: Optional[str] = field(default="facebook/opt-125m")
    llm_freeze: bool = field(default=False) # freeze LLM except embedding layer
    llm_emb_freeze: bool = field(default=False)
    llm_head_freeze: bool = field(default=False)
    sllm_weight_path: Optional[str] = field(default=None)
    use_flash_attn: bool = field(default=False)
    lora_rank: Optional[int] = field(default=-1)

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    data_split_train: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    data_split_eval: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    audio_normalize: bool = field(
        default=False,
        metadata={"help": "Normalize audio"}
    )
    source_lang: str = field(
        default="English",
        metadata={"help": "Source language name"}
    )
    target_lang: str = field(
        default="Spanish",
        metadata={"help": "Target language name"}
    )
    trajectory: int = field(
        default=0,
        metadata={"help": "0: offline, 1: offline instruct, 2: trajectory, 3: trajectory with instruct format"}
    )
    trajectory_perturb: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0],
        metadata={"help": "Perturbation for trajectory"}
    )
    trajectory_step_size: int = field(
        default=1,
        metadata={"help": "Step size for trajectory"}
    )
    trajectory_max_multiplier: int = field(
        default=1,
        metadata={"help": "Maximum multiplier for trajectory"}
    )
    preference_optimization_max_multiplier: int = field(
        default=1,
        metadata={"help": "Maximum multiplier for preference optimization"}
    )
    trajectory_prob_aug: float = field(
        default=0.0,
        metadata={"help": "Probability of augmentation for trajectory"}
    )
                            
@dataclass
class TrainingArguments:
    seed: int = field(default=1)
    stage: int = field(default=1)
    rdrop: float = field(default=0.)
    text_weight: float = field(default=0.)
    train_bsz: int = field(default=8) # in terms of number of frames
    eval_bsz: int = field(default=8) # in terms of number of frames
    bsz_sent: int = field(default=3) # in terms of number of sentences
    learning_rate: float = field(default=2e-4)
    scheduler: str = field(default="cosine")
    min_learning_rate: float = field(default=0.)
    weight_decay: float = field(default=0.)
    warmup_steps: int = field(default=400)
    run_name: str = field(default=None)

    cpo_beta: float = field(default=0.0)

    n_device: int = field(default=1)
    deepspeed_stage: int = field(default=2)
    deepspeed_offload: bool = field(default=False)
    deepspeed_bucket_size: int = field(default=int(2e8))
    
    precision: str = field(default="bf16-mixed")
    max_epochs: int = field(default=1)
    grad_acc_steps: int = field(default=1)
    clip_norm: float = field(default=1.)
    save_dir: str = field(default=None)
    save_step: int = field(default=1000)
    log_step: int = field(default=1)
    eval_step: int = field(default=1)
    debug_mode: bool = field(default=False)

    profile: str = field(default=None, metadata={"help": "Profile to use"})

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    parser = transformers.HfArgumentParser(
        (SpeechEncoderArguments, ModelArguments, DataArguments, TrainingArguments))
    speech_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    torch.set_float32_matmul_precision('high')

    # Set seed before initializing model.
    set_seed(training_args.seed) 

    model_class = MODEL_CLASSES[model_args.model_type]  
    model_lightning = model_class(
        speech_args=speech_args,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        lr=training_args.learning_rate,
        warmup_updates=training_args.warmup_steps,
        min_lr=0.,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=training_args.save_dir,
        # save_on_train_epoch_end=True,
        every_n_train_steps=training_args.save_step,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )

    wandb_logger = WandbLogger(
        name=training_args.run_name,
        # log_model="all"
    )

    # Use standard DDP strategy (no DeepSpeed)
    # Some parameters (e.g., frozen heads/encoders) may not participate in loss every step.
    # Enable find_unused_parameters to avoid reducer errors.
    strategy = DDPStrategy(find_unused_parameters=True)
    # strategy = FSDPStrategy(
    #     sharding_strategy=training_args.sharding,
    #     state_dict_type="sharded"
    # )

    profiler = None
    if training_args.profile == "advanced":
        profiler = AdvancedProfiler(
            filename="profile"
        )
    elif training_args.profile == "simple":
        profiler = SimpleProfiler(
            filename="profile"
        )
    elif training_args.profile == "pytorch":
        profiler = PyTorchProfiler(
            filename="profile"
        )

    trainer = L.Trainer(
        accelerator='gpu',
        devices=training_args.n_device,
        strategy=strategy,
        precision=training_args.precision,
        max_steps=-1,
        max_epochs=training_args.max_epochs,
        accumulate_grad_batches=training_args.grad_acc_steps,
        gradient_clip_val=training_args.clip_norm,
        use_distributed_sampler=False,
        default_root_dir=training_args.save_dir,
        log_every_n_steps=training_args.log_step,
        val_check_interval=training_args.eval_step,
        logger=wandb_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        fast_dev_run=training_args.debug_mode,
        # enable_checkpointing=False,
        profiler=profiler
    )

    # start training
    if os.path.exists(training_args.save_dir) and len(os.listdir(training_args.save_dir)) >= 1:
        ckpt_path = os.path.join(training_args.save_dir, 'last.ckpt', 'checkpoint')
        trainer.fit(model_lightning, ckpt_path=ckpt_path)
    else:
        trainer.fit(model_lightning)

    # trainer.save_checkpoint(training_args.save_dir, weights_only=True)


if __name__ == "__main__":
    train()