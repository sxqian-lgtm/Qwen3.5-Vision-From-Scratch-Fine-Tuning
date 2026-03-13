#!/usr/bin/env python
# coding: utf-8

# # 04 - Fine-Tuning with Selective Layers
# Freeze stable blocks and train only selected modules (for example, cross-attention + LM head).

# In[ ]:


from setup_01 import *
from data_02 import data_load
import torch
from model_03 import Qwen35Vision
import os
import math
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
def init_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, rank, world_size, device


# In[ ]:


TRAIN_CFG = CONFIG.get("train_session", {})

class TrainSessionManager:
    def __init__(self,dataset_size=None,model_size=None, resume_training=None,state_dir=None,
            gpu_ids=None,use_ddp=False,local_rank=0,rank=0,world_size=1,device=None):
            self.dataset_size = dataset_size if dataset_size is not None else TRAIN_CFG.get("dataset_size", 2)
            self.model_size = model_size if model_size is not None else TRAIN_CFG.get("model_size", "0.8b")
            self.resume_training = resume_training if resume_training is not None else TRAIN_CFG.get("resume_training", True)

            self.first_model_filename = TRAIN_CFG.get("first_model_filename", "first")
            self.latest_model_filename = TRAIN_CFG.get("latest_model_filename", "latest")
            base_state_dir = state_dir if state_dir is not None else TRAIN_CFG.get("state_dir", "outputs/train_logs")

            self.gpu_ids = gpu_ids if gpu_ids is not None else TRAIN_CFG.get("gpu_ids", None)

            self.use_ddp = use_ddp
            self.local_rank = local_rank
            self.rank = rank
            self.world_size = world_size

            self.model_tag = str(self.model_size).replace("/", "_")
            self.state_dir = os.path.join(base_state_dir, self.model_tag)
            os.makedirs(self.state_dir, exist_ok=True)
            self.latest_state_path = os.path.join(self.state_dir, f"latest_train_state_{self.model_tag}.pt")

            set_seed(CONFIG["seed"] + self.rank)
            self.Qwen = Qwen35Vision(model_size=self.model_size,device=device)
            self.dataset = None
            self.optimizer = None
            self.job_index = 0

    def _model_store_dir(self, filename):
        safe_model_id = self.Qwen.MODEL_ID.replace("/", "_")
        safe_filename = str(filename).replace("/", "_")
        return os.path.join(self.Qwen.save_dir, f"{safe_model_id}_{safe_filename}")

    def setup(self, lora_r=None, lora_alpha=None, lora_dropout=None):
        lora_cfg = TRAIN_CFG.get("lora", {})
        lora_r = lora_r if lora_r is not None else lora_cfg.get("r", 8)
        lora_alpha = lora_alpha if lora_alpha is not None else lora_cfg.get("alpha", 16)
        lora_dropout = lora_dropout if lora_dropout is not None else lora_cfg.get("dropout", 0.05)

        latest_model_dir = self._model_store_dir(self.latest_model_filename)

        if self.resume_training and os.path.exists(self.latest_state_path) and os.path.isdir(latest_model_dir):
            print(f"Resuming from state: {self.latest_state_path}")
            self.Qwen.model_load(Install=False, filename=self.latest_model_filename)
            print("Loaded checkpoint model with existing LoRA/PEFT. Skip enable_lora().")
        else:
            print("No previous state found. Starting from base model.")
            self.Qwen.model_load(Install=True)
            self.Qwen.enable_lora(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        # Keep trainable parameters in float32 for stable AMP training.
        # Base frozen weights can stay in fp16 to save memory.
        for name, p in self.Qwen.model.named_parameters():
            if p.requires_grad and p.data.is_floating_point():
                p.data = p.data.float()
                

        if self.use_ddp:
            self.Qwen.device = torch.device(f"cuda:{self.local_rank}")
            print(f"[setup] rank={self.rank}, set Qwen.device={self.Qwen.device}")

            self.Qwen.model = DDP(
                self.Qwen.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            if self.rank == 0:
                print(
                    f"DDP enabled on rank {self.rank}, "
                    f"local_rank {self.local_rank}, world_size {self.world_size}"
                )
        else:
            print("Running in single-device mode.")

        # Create optimizer after model / DDP wrapping
        self.optimizer = torch.optim.AdamW(
            (p for p in self.Qwen.model.parameters() if p.requires_grad),
            lr=CONFIG["training"]["learning_rate"],
            weight_decay=CONFIG["training"]["weight_decay"],
        )

        if self.resume_training and os.path.exists(self.latest_state_path):
            state_obj = torch.load(self.latest_state_path, map_location="cpu")
            state_model_id = state_obj.get("model_id")
            current_model_id = self.Qwen.MODEL_ID

            if state_model_id is not None and state_model_id != current_model_id:
                print(
                    f"Resume state model mismatch; skip optimizer/history load. "
                    f"state={state_model_id}, current={current_model_id}"
                )
            else:
                # Try to restore optimizer state for seamless resume
                if "optimizer_state_dict" in state_obj:
                    try:
                        self.optimizer.load_state_dict(state_obj["optimizer_state_dict"])
                        print("Optimizer state restored successfully.")
                    except Exception as e:
                        print(f"Warning: failed to load optimizer state. Using fresh optimizer. Error: {e}")

                self.Qwen.train_history = state_obj.get("train_history", self.Qwen.train_history)
                self.Qwen.train_eval_history = state_obj.get(
                    "train_eval_history",
                    self.Qwen.train_eval_history
                )
                self.job_index = int(state_obj.get("job_index", 0)) + 1

        print(f"Model and optimizer are ready. Job index: {self.job_index}")

    def run_and_save(self, num_epochs=None):
        num_epochs = num_epochs if num_epochs is not None else TRAIN_CFG.get("num_epochs", 1)

        train_result = self.Qwen.train_run(
            dataset=self.dataset,
            optimizer=self.optimizer,
            num_epochs=num_epochs,
            sampler_rank=self.rank,
            sampler_world_size=self.world_size,
            use_ddp=self.use_ddp,
        )

        if self.rank != 0:
            return train_result

        print("Train test finished.")
        print(train_result)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Keep only first and latest model snapshots.
        first_model_dir = self._model_store_dir(self.first_model_filename)
        if not os.path.isdir(first_model_dir):
            self.Qwen.model_save(self.first_model_filename)
            print(f"Saved first model snapshot: {first_model_dir}")

        self.Qwen.model_save(self.latest_model_filename)

        # Save raw JSON
        result_json_path = os.path.join(self.state_dir, f"train_result_{run_id}.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(train_result, f, indent=2, default=str)

        # Save epoch metrics CSV
        csv_path = os.path.join(self.state_dir, f"metrics_by_epoch_{run_id}.csv")
        train_loss = train_result.get("train_loss", [])
        eval_metrics = train_result.get("eval_metrics", {})
        val_loss = eval_metrics.get("val_loss", [])
        token_acc = eval_metrics.get("token_acc", [])
        perplexity = eval_metrics.get("perplexity", [])
        max_len = max(len(train_loss), len(val_loss), len(token_acc), len(perplexity), 1)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "token_acc", "perplexity"])
            for i in range(max_len):
                writer.writerow([
                    i + 1,
                    train_loss[i] if i < len(train_loss) else "",
                    val_loss[i] if i < len(val_loss) else "",
                    token_acc[i] if i < len(token_acc) else "",
                    perplexity[i] if i < len(perplexity) else "",
                ])

        # Append cross-run summary
        global_csv = os.path.join(self.state_dir, "all_runs_summary.csv")
        global_exists = os.path.exists(global_csv)
        with open(global_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not global_exists:
                writer.writerow([
                    "run_id", "job_index", "dataset_size",
                    "last_train_loss", "last_val_loss",
                    "last_token_acc", "last_perplexity"
                ])
            writer.writerow([
                run_id,
                self.job_index,
                len(self.dataset),
                train_loss[-1] if len(train_loss) > 0 else "",
                val_loss[-1] if len(val_loss) > 0 else "",
                token_acc[-1] if len(token_acc) > 0 else "",
                perplexity[-1] if len(perplexity) > 0 else "",
            ])

        # Save loss figure
        fig_path = None
        if len(train_loss) > 0 or len(val_loss) > 0:
            plt.figure(figsize=(8, 5))
            if len(train_loss) > 0:
                plt.plot(range(1, len(train_loss) + 1), train_loss, marker="o", label="train_loss")
            if len(val_loss) > 0:
                plt.plot(range(1, len(val_loss) + 1), val_loss, marker="s", label="val_loss")
            plt.title("Training Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(alpha=0.3)
            plt.legend()
            fig_path = os.path.join(self.state_dir, f"loss_curve_{run_id}.png")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

        # Save resume state
        resume_state = {
            "job_index": self.job_index,
            "run_id": run_id,
            "model_id": self.Qwen.MODEL_ID,
            "model_size": self.model_size,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_history": self.Qwen.train_history,
            "train_eval_history": self.Qwen.train_eval_history,
        }
        state_path = os.path.join(self.state_dir, f"train_state_{run_id}.pt")
        torch.save(resume_state, state_path)
        torch.save(resume_state, self.latest_state_path)

        # Save metadata
        meta = {
            "run_id": run_id,
            "job_index": self.job_index,
            "model_id": self.Qwen.MODEL_ID,
            "device": str(self.Qwen.device),
            "dataset_size": len(self.dataset),
            "num_epochs": num_epochs,
            "files": {
                "train_result_json": result_json_path,
                "metrics_csv": csv_path,
                "global_summary_csv": global_csv,
                "loss_curve_png": fig_path,
                "resume_state": state_path,
                "latest_resume_state": self.latest_state_path,
            },
        }
        meta_path = os.path.join(self.state_dir, f"run_meta_{run_id}.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        if fig_path is not None:
            print(f"- PNG:  {fig_path}")
        print(f"- STATE: {state_path}")
        print(f"- LATEST: {self.latest_state_path}")
        print(f"- META: {meta_path}")

        return train_result


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen3.5 vision model with DDP")
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--model-size", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    local_rank, rank, world_size,device_cuda = init_ddp()
    print(f"[main] local_rank={local_rank}, rank={rank}, world_size={world_size}, device_cuda={device_cuda!r}", flush=True)
    assert str(device_cuda) == f"cuda:{local_rank}", f"device_cuda wrong: {device_cuda}"

    session = TrainSessionManager(
        dataset_size=args.dataset_size,
        model_size=args.model_size,
        resume_training=not args.no_resume,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        use_ddp=True,
        device=device_cuda
    )
    session.setup()
    #self.dataset = data_load().select(range(self.dataset_size))
    full_dataset = data_load().select(range(session.dataset_size))
    chunk_size = args.chunk_size if args.chunk_size is not None else len(full_dataset)

    start = session.job_index * chunk_size
    end = start + chunk_size

    if start >= len(full_dataset):
        if rank == 0:
            print(f"All dataset chunks finished. start={start}, total={len(full_dataset)}")
        dist.destroy_process_group()
        sys.exit(0)

    print(f"[dataset chunk] start={start}, end={min(end, len(full_dataset))}, chunk_size={chunk_size}")
    total_chunks = math.ceil(len(full_dataset) / chunk_size)
    print(f"[chunk info] total_samples={len(full_dataset)}, chunk_size={chunk_size}, total_chunks={total_chunks}")
    session.dataset = full_dataset.select(range(start, min(end, len(full_dataset))))
    
    train_result = session.run_and_save(num_epochs=args.num_epochs)
    dist.destroy_process_group()


# In[1]:




