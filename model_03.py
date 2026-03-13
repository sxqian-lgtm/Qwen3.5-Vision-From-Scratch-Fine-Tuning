#!/usr/bin/env python
# coding: utf-8

# # 03 - Model Architecture from Scratch
# Implement the vision encoder, text decoder, cross-attention, and LM head.

# In[1]:


from setup_01 import *
from data_02 import data_load
import os
import torch
import math
import time
from transformers import AutoModelForImageTextToText
from transformers import AutoTokenizer
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


# In[2]:


class Qwen35Vision():
    def __init__(self, model_size='0.8b',device=None):
        self.device = device if device is not None else torch.device(CONFIG["device"])
        # if os.environ.get("LOCAL_RANK") is not None:
        #     expect = f"cuda:{os.environ['LOCAL_RANK']}"
        #     assert str(self.device) == expect, f"Expected {expect}, got {self.device}"
        self.dtype = CONFIG["dtype"]
        self.MODEL_ID = CONFIG['models'][model_size]
        self.save_dir = "checkpoints"
        os.makedirs(self.save_dir, exist_ok=True)
        self.tokenizer = self.tokenizer_access(model_key=model_size)
        self.processor = self.processor_access(model_key=model_size)
        self.is_lora_enabled = False
        self.train_history = []
        self.train_eval_history = {
            "val_loss": [], 
            "token_acc": [], 
            "perplexity": []}
        self.eval_metrics = {}
        print("Qwen init device:", self.device)
        print("LOCAL_RANK env:", os.environ.get("LOCAL_RANK"))

    def tokenizer_access(self, model_key="0.8b"):
        repo_id = CONFIG["models"][model_key]
        token = os.getenv(CONFIG["hf"]["token_env"])
        print(f"Checking tokenizer access for: {repo_id}")
        print("HF token found in env:", bool(token))
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                token=token,
                trust_remote_code=CONFIG["hf"]["trust_remote_code"],
            )
            print("Tokenizer loaded successfully.")
            print("vocab_size:", tokenizer.vocab_size)
            return tokenizer
        except Exception as exc:
            print("Tokenizer load failed.")
            print("Reason:", exc)
            print("Hint: run `huggingface-cli login` or set HF_TOKEN env variable.")
            return None

    def processor_access(self, model_key="0.8b"):
        repo_id = CONFIG["models"][model_key]
        token = os.getenv(CONFIG["hf"]["token_env"])
        print(f"Checking processor access for: {repo_id}")
        print("HF token found in env:", bool(token))
        try:
            processor = AutoProcessor.from_pretrained(
                repo_id,
                token=token,
                trust_remote_code=CONFIG["hf"]["trust_remote_code"],
            )
            print("Processor loaded successfully.")
            return processor
        except Exception as exc:
            print("Processor load failed.")
            print("Reason:", exc)
            print("Hint: run `huggingface-cli login` or set HF_TOKEN env variable.")
            return None

    def data_transform(
        self,
        dataset,
        max_text_len=CONFIG["data"]["max_text_len"],
        prompt_max_len=CONFIG["data"]["image_token_len"],
        image_size_height=CONFIG["data"]["image_size_height"],
        image_size_width=CONFIG["data"]["image_size_width"],
        prompt="OCR Latex."
    ):
        """
        Build multimodal training samples for VL model.

        Strategy:
        1. Use processor to build image + prompt input.
        2. Use tokenizer to build target text.
        3. Fill target tokens into the free padded positions of prompt sequence.
        4. Only target positions contribute to loss (labels != -100).
        """

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []
        pixel_attention_mask_list = []
        image_grid_thw_list = []

        data_len = len(dataset)
        print(f"Transforming dataset of size: {data_len}")
        print(f"Using max_text_len: {max_text_len}")
        print(f"Using prompt_max_len: {prompt_max_len}")
        print(f"Using image_size: {image_size_height}x{image_size_width}")

        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        eos_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        print(f"Using image token: {image_token}")

        for i, sample in enumerate(dataset):
            target_text = sample["text"]
            image = sample["image"]

            if image is None:
                raise ValueError(f"Sample {i} has no image.")

            if hasattr(image, "convert"):
                image = image.convert("RGB")

            if hasattr(image, "resize"):
                # Keep your wide LaTeX image resize rule
                image = image.resize((image_size_width, image_size_height))

            prompt_text = f"{image_token}\n{prompt}"

            # Build prompt + image inputs
            model_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=prompt_max_len,
            )

            # Build target text tokens
            target_out = self.tokenizer(
                text=target_text,
                truncation=True,
                padding=False,   # do not pad here
                max_length=max_text_len,
                return_tensors="pt",
            )

            input_ids = model_inputs["input_ids"].squeeze(0)
            attention_mask = model_inputs["attention_mask"].squeeze(0)
            pixel_values = model_inputs["pixel_values"].squeeze(0)

            if "image_grid_thw" not in model_inputs:
                raise RuntimeError("processor output missing image_grid_thw; cannot run VL model reliably.")

            image_grid_thw = model_inputs["image_grid_thw"].squeeze(0)

            target_ids = target_out["input_ids"].squeeze(0)

            # Ensure EOS at end
            if eos_id is not None and target_ids.numel() > 0 and target_ids[-1].item() != eos_id:
                eos_tensor = torch.tensor([eos_id], dtype=target_ids.dtype)
                target_ids = torch.cat([target_ids, eos_tensor], dim=0)

            # Prepare train tensors
            train_input_ids = input_ids.clone()
            train_attention_mask = attention_mask.clone()
            train_labels = torch.full_like(train_input_ids, -100)

            occupied_mask = train_attention_mask.bool()
            free_positions = torch.where(~occupied_mask)[0]

            available_len = int(free_positions.numel())
            target_len = int(target_ids.numel())

            if target_len > available_len:
                raise ValueError(
                    f"Sample {i} too long: target_len={target_len}, available_len={available_len}. "
                    f"Increase prompt_max_len or reduce max_text_len."
                )

            if target_len > 0:
                target_positions = free_positions[:target_len]
                train_input_ids[target_positions] = target_ids
                train_attention_mask[target_positions] = 1
                train_labels[target_positions] = target_ids

            input_ids_list.append(train_input_ids)
            attention_mask_list.append(train_attention_mask)
            labels_list.append(train_labels)
            pixel_values_list.append(pixel_values)
            image_grid_thw_list.append(image_grid_thw)

        # Pad visual tokens to same length across samples
        max_visual_tokens = max(pv.shape[0] for pv in pixel_values_list)
        hidden_dim = pixel_values_list[0].shape[1]

        padded_pixel_values = []
        pixel_attention_mask = []

        for pv in pixel_values_list:
            cur_len = pv.shape[0]

            if cur_len < max_visual_tokens:
                pad = torch.zeros(
                    (max_visual_tokens - cur_len, hidden_dim),
                    dtype=pv.dtype
                )
                pv = torch.cat([pv, pad], dim=0)

            padded_pixel_values.append(pv)

            mask = torch.zeros(max_visual_tokens, dtype=torch.long)
            mask[:cur_len] = 1
            pixel_attention_mask.append(mask)

        output = {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
            "pixel_values": torch.stack(padded_pixel_values),
            "pixel_attention_mask": torch.stack(pixel_attention_mask),
            "image_grid_thw": torch.stack(image_grid_thw_list),
        }

        print("Transform complete.")
        print("input_ids shape:", output["input_ids"].shape, output["input_ids"].dtype)
        print("attention_mask shape:", output["attention_mask"].shape, output["attention_mask"].dtype)
        print("labels shape:", output["labels"].shape, output["labels"].dtype)
        print("pixel_values shape:", output["pixel_values"].shape, output["pixel_values"].dtype)
        print("pixel_attention_mask shape:", output["pixel_attention_mask"].shape, output["pixel_attention_mask"].dtype)
        print("image_grid_thw shape:", output["image_grid_thw"].shape, output["image_grid_thw"].dtype)

        return output

    def data_transform_gen(self, dataset, max_length=None, image_token_len=None, image_size_height=None, image_size_width=None, prompt="Describe this image."):
        """Build generation-specific inputs and labels only."""
        if max_length is None:
            max_length = CONFIG["data"]["max_text_len"]
        if image_token_len is None:
            image_token_len = CONFIG["data"].get("image_token_len", 2048)
        if image_size_height is None:
            image_size_height = CONFIG["data"].get("image_size_height", CONFIG["data"].get("image_size", 224))
        if image_size_width is None:
            image_size_width = CONFIG["data"].get("image_size_width", CONFIG["data"].get("image_size", 224))

        gen_input_ids_list = []
        gen_attention_mask_list = []
        gen_labels_list = []
        gen_label_mask_list = []
        gen_pixel_values_list = []
        gen_image_grid_thw_list = []

        image_token = getattr(self.processor, "image_token", "<|image_pad|>")
        eos_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None

        for sample in dataset:
            target_text = sample["text"]
            image = sample["image"]
            if image is not None and hasattr(image, "resize"):
                image = image.convert("RGB")
                image = image.resize((image_size_width, image_size_height))

            prompt_text = f"{image_token}\n{prompt}"
            input_out = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=image_token_len,
            )
            output_out = self.tokenizer(
                text=target_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

            gen_input_ids = input_out["input_ids"].squeeze(0)
            gen_attention_mask = input_out["attention_mask"].squeeze(0)
            gen_pixel_values = input_out["pixel_values"].squeeze(0)
            if "image_grid_thw" not in input_out:
                raise RuntimeError("processor output missing image_grid_thw; cannot run VL model reliably.")
            gen_image_grid_thw = input_out["image_grid_thw"].squeeze(0)
            output_ids = output_out["input_ids"].squeeze(0)
            output_mask_text = output_out["attention_mask"].squeeze(0)

            target_ids = output_ids[output_mask_text.bool()]
            if eos_id is not None and target_ids.numel() > 0 and target_ids[-1].item() != eos_id:
                eos_tensor = torch.tensor([eos_id], device=target_ids.device, dtype=target_ids.dtype)
                target_ids = torch.cat([target_ids, eos_tensor], dim=0)

            gen_labels = torch.full_like(output_ids, -100)
            gen_label_mask = torch.zeros_like(output_mask_text)
            gen_target_len = int(min(target_ids.numel(), output_ids.numel()))
            if gen_target_len > 0:
                gen_labels[:gen_target_len] = target_ids[:gen_target_len]
                gen_label_mask[:gen_target_len] = 1

            gen_input_ids_list.append(gen_input_ids)
            gen_attention_mask_list.append(gen_attention_mask)
            gen_labels_list.append(gen_labels)
            gen_label_mask_list.append(gen_label_mask)
            gen_pixel_values_list.append(gen_pixel_values)
            gen_image_grid_thw_list.append(gen_image_grid_thw)

        max_visual_tokens = max(pv.shape[0] for pv in gen_pixel_values_list)
        hidden_dim = gen_pixel_values_list[0].shape[1]
        padded_gen_pixel_values = []
        gen_pixel_attention_mask = []
        for pv in gen_pixel_values_list:
            cur_len = pv.shape[0]
            if cur_len < max_visual_tokens:
                pad = torch.zeros((max_visual_tokens - cur_len, hidden_dim), dtype=pv.dtype)
                pv = torch.cat([pv, pad], dim=0)
            padded_gen_pixel_values.append(pv)
            mask = torch.zeros(max_visual_tokens, dtype=torch.long)
            mask[:cur_len] = 1
            gen_pixel_attention_mask.append(mask)

        return {
            "model_gen_input": torch.stack(gen_input_ids_list),
            "model_gen_mask": torch.stack(gen_attention_mask_list),
            "model_gen_label": torch.stack(gen_labels_list),
            "model_gen_label_mask": torch.stack(gen_label_mask_list),
            "model_gen_pixel_values": torch.stack(padded_gen_pixel_values),
            "model_gen_pixel_attention_mask": torch.stack(gen_pixel_attention_mask),
            "model_gen_image_grid_thw": torch.stack(gen_image_grid_thw_list),
        }

    def data_divide(self, dataset, test_size=0.2, seed=42):
        split_ds = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        train_raw = split_ds["train"]
        test_raw = split_ds["test"]

        dataset_train = self.data_transform(train_raw)
        dataset_test = self.data_transform(test_raw)
        return dataset_train, dataset_test

    def data_divide_gen(self, dataset, test_size=0.2, seed=42):
        split_ds = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        test_raw = split_ds["test"]

        dataset_gen_test = self.data_transform_gen(test_raw)
        return dataset_gen_test

    def enable_lora(self, r=16, alpha=32, dropout=0.05):
        if not hasattr(self, "model"):
            raise RuntimeError("Model is not loaded. Call model_load(...) first.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.enable_input_require_grads()
        self.is_lora_enabled = True
        self.model.print_trainable_parameters()

    def train_run(
        self,
        dataset,
        optimizer,
        num_epochs=CONFIG["training"]["num_epochs"]["test1"],
        test_size=0.1,
        seed=CONFIG["seed"],
        sampler_rank=0,
        sampler_world_size=1,
        use_ddp=False,
    ):
        if not hasattr(self, "model"):
            raise RuntimeError("Model is not loaded. Call model_load(...) first.")

        # 读取 batch size
        batch_size = CONFIG["training"].get("batch_size", 1)

        # 划分训练/验证集
        dataset_train, dataset_test = self.data_divide(
            dataset, test_size=test_size, seed=seed
        )

        # 构造 train TensorDataset
        train_dataset = TensorDataset(
            dataset_train["input_ids"],
            dataset_train["attention_mask"],
            dataset_train["pixel_values"],
            dataset_train["pixel_attention_mask"],
            dataset_train["image_grid_thw"],
            dataset_train["labels"],
        )

        # 构造 eval TensorDataset
        eval_dataset = TensorDataset(
            dataset_test["input_ids"],
            dataset_test["attention_mask"],
            dataset_test["pixel_values"],
            dataset_test["pixel_attention_mask"],
            dataset_test["image_grid_thw"],
            dataset_test["labels"],
        )

        # DDP sampler
        if use_ddp:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=sampler_world_size,
                rank=sampler_rank,
                shuffle=True,
            )
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=sampler_world_size,
                rank=sampler_rank,
                shuffle=False,
            )
        else:
            train_sampler = None
            eval_sampler = None

        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=eval_sampler,
        )

        # training loop
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.device.type == "cuda")
        )
        
        for epoch in range(num_epochs):
            log_interval = 100   # print every 10 batches
            step = 0
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            total_loss = 0.0
            self.model.train()
            stop_training = False

            for (
                input_ids,
                attention_mask,
                pixel_values,
                pixel_attention_mask,
                image_grid_thw,
                labels,
            ) in train_loader:
                
                step_start = time.time()

                optimizer.zero_grad(set_to_none=True)

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                pixel_values = pixel_values.to(self.device)
                pixel_attention_mask = pixel_attention_mask.to(self.device)
                image_grid_thw = image_grid_thw.to(self.device)
                labels = labels.to(self.device)

                model_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "labels": labels,
                }

                with torch.amp.autocast(
                    "cuda",
                    enabled=(self.device.type == "cuda"),
                    dtype=torch.float16,
                ):
                    try:
                        outputs = self.model(
                            **model_kwargs,
                            pixel_attention_mask=pixel_attention_mask,
                        )
                    except TypeError:
                        outputs = self.model(**model_kwargs)

                    loss = outputs.loss

                if not torch.isfinite(loss):
                    print("WARNING: loss is NaN/Inf, skip this batch.")
                    stop_training = True
                    break

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()

                bad_param_found = False
                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.data.is_floating_point():
                        if not torch.isfinite(p.data).all():
                            print(f"Non-finite parameter detected after step: {name}")
                            bad_param_found = True
                            break

                if bad_param_found:
                    print("Stopping because parameters became NaN/Inf after optimizer.step()")
                    stop_training = True
                    break

                total_loss += loss.item()
                step += 1
                
                if step % log_interval == 0:
                    step_time = time.time() - step_start
                    print(
                        f"step {step} | "
                        f"loss {loss.item():.4f} | "
                        f"time {step_time:.3f}s",
                        flush=True
                    )

            if stop_training:
                raise RuntimeError("Training stopped because loss or parameters became NaN/Inf.")

            epoch_train_loss = total_loss / max(len(train_loader), 1)
            self.train_history.append(epoch_train_loss)

            metrics = self.eval_run(
                eval_loader=eval_loader,
                use_ddp=use_ddp,
                sampler_rank=sampler_rank,
                sampler_world_size=sampler_world_size,
            )

            self.train_eval_history["val_loss"].append(metrics["val_loss"])
            self.train_eval_history["token_acc"].append(metrics["token_acc"])
            self.train_eval_history["perplexity"].append(metrics["perplexity"])

        return {
            "train_loss": self.train_history,
            "eval_metrics": self.train_eval_history,
        }

    def eval_run(
        self,
        eval_loader,
        use_ddp=False,
        sampler_rank=0,
        sampler_world_size=1,
    ):
        if not hasattr(self, "model"):
            raise RuntimeError("Model is not loaded. Call model_load(...) first.")

        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, (
                input_ids,
                attention_mask,
                pixel_values,
                pixel_attention_mask,
                image_grid_thw,
                labels,
            ) in enumerate(eval_loader, start=1):
                # print(f"Evaluating batch {batch_idx}/{len(eval_loader)}", end="\r")

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                pixel_values = pixel_values.to(self.device)
                pixel_attention_mask = pixel_attention_mask.to(self.device)
                image_grid_thw = image_grid_thw.to(self.device)
                labels = labels.to(self.device)

                model_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "labels": labels,
                }

                try:
                    preds = self.model(
                        **model_kwargs,
                        pixel_attention_mask=pixel_attention_mask,
                    )
                except TypeError:
                    preds = self.model(**model_kwargs)

                loss = preds.loss
                logits = preds.logits
                
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                shift_preds = torch.argmax(shift_logits, dim=-1)
                valid_mask = shift_labels != -100

                if valid_mask.any():
                    correct_tokens += (shift_preds[valid_mask] == shift_labels[valid_mask]).sum().item()
                    total_tokens += valid_mask.sum().item()

                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / max(num_batches, 1)
        token_acc = correct_tokens / max(total_tokens, 1)

        # 防止 overflow
        perplexity = math.exp(avg_val_loss) if avg_val_loss < 20 else float("inf")

        metrics = {
            "val_loss": avg_val_loss,
            "token_acc": token_acc,
            "perplexity": perplexity,
        }
        self.eval_metrics = metrics

        # print()
        # print(f"Eval loss: {avg_val_loss:.6f}")
        # print(f"Token accuracy: {token_acc:.4f}")
        # print(f"Perplexity: {perplexity:.4f}")

        return metrics
    
    def gen_eval_run(self, dataset_gen_test, max_new_tokens=None):
        if not hasattr(self, "model"):
            raise RuntimeError("Model is not loaded. Call model_load(...) first.")

        self.model.eval()
        if max_new_tokens is None:
            max_new_tokens = CONFIG["data"]["max_text_len"]
        n_samples = dataset_gen_test["model_gen_input"].shape[0]

        correct_tokens = 0
        total_tokens = 0
        exact_match = 0

        with torch.no_grad():
            for i in range(n_samples):
                print(f"Gen evaluating sample {i + 1}/{n_samples}", end="\r")
                ##i:i+1 to keep batch dimension for generation
                input_ids = dataset_gen_test["model_gen_input"][i:i+1].to(self.device)
                attention_mask = dataset_gen_test["model_gen_mask"][i:i+1].to(self.device)
                labels = dataset_gen_test["model_gen_label"][i:i+1].to(self.device)
                label_mask = dataset_gen_test["model_gen_label_mask"][i:i+1].to(self.device)

                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=dataset_gen_test["model_gen_pixel_values"][i:i+1].to(self.device),
                    image_grid_thw=dataset_gen_test["model_gen_image_grid_thw"][i:i+1].to(self.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                pred_new_tokens = generated[:, input_ids.shape[1]:]
                target_len = labels.shape[1]

                if pred_new_tokens.shape[1] < target_len:
                    pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                    pad = torch.full((pred_new_tokens.shape[0], target_len - pred_new_tokens.shape[1]), pad_id, device=pred_new_tokens.device, dtype=pred_new_tokens.dtype)
                    pred_new_tokens = torch.cat([pred_new_tokens, pad], dim=1)
                elif pred_new_tokens.shape[1] > target_len:
                    pred_new_tokens = pred_new_tokens[:, :target_len]

                valid_mask = label_mask.bool() & (labels != -100)
                correct_tokens += (pred_new_tokens[valid_mask] == labels[valid_mask]).sum().item()
                total_tokens += valid_mask.sum().item()

                sample_mask = valid_mask[0]
                if sample_mask.any() and torch.equal(pred_new_tokens[0][sample_mask], labels[0][sample_mask]):
                        exact_match += 1

        token_acc = correct_tokens / max(total_tokens, 1)
        exact_match_acc = exact_match / max(n_samples, 1)

        metrics = {
            "gen_token_acc": token_acc,
            "gen_exact_match": exact_match_acc,
            "gen_total_tokens": total_tokens,
        }
        return metrics

    def _model_store_dir(self, filename):
        safe_model_id = self.MODEL_ID.replace("/", "_")
        safe_filename = str(filename).replace("/", "_")
        return os.path.join(self.save_dir, f"{safe_model_id}_{safe_filename}")

    def model_save(self, filename="latest"):
        if not hasattr(self, 'model'):
            print("No model to save.")
            return
        save_path = self._model_store_dir(filename)
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        base_model.save_pretrained(
            save_path,
            safe_serialization=True,
            max_shard_size="1GB",
        )
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)

        if hasattr(self, "processor") and self.processor is not None:
            self.processor.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    def model_load(self, Install=False, filename="latest"):
        if Install:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL_ID,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()
            self.model.eval()

            self.num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Loaded (VL): {self.MODEL_ID}")
            
        else:
            load_path = self._model_store_dir(filename)
            self.model = AutoModelForImageTextToText.from_pretrained(
                load_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()
            self.model.eval()
            print(f"Model loaded (VL) from {load_path}")
            


# In[6]:


if __name__ == "__main__":
    set_seed(CONFIG['seed'])
    model3508b = Qwen35Vision()
    model3508b.model_load(Install=True)
    model3508b.enable_lora(r=16, alpha=32, dropout=0.05)

    dataset = data_load()
    subset = dataset.select(range(100))

    # 这里建议和你实际训练时保持一致
    prompt = "Describe this image."
    image_token = getattr(model3508b.processor, "image_token", "<|image_pad|>")

    tokenizer = model3508b.tokenizer_access(model_key='0.8b')

    text_lengths = []
    prompt_lengths = []
    required_lengths = []
    vision_token_lengths = []

    # 先查前 1000 或 2000 条就够了，没必要一上来 10000
    inspect_n = min(21000, len(dataset))

    for i, ex in enumerate(dataset.select(range(inspect_n))):
        text = ex["text"]
        image = ex["image"]

        # ---------- 1) 纯 target_text 的长度 ----------
        target_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        target_len = len(target_ids)
        text_lengths.append(target_len)

        # ---------- 2) processor 后 prompt 已占长度 ----------
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        if hasattr(image, "resize"):
            image = image.resize((
                CONFIG["data"]["image_size_width"],
                CONFIG["data"]["image_size_height"]
            ))

        prompt_text = f"{image_token}\n{prompt}"

        model_inputs = model3508b.processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=CONFIG["data"]["image_token_len"],   # 你现在代码里的 prompt_max_len
        )

        # 语言侧已经占掉的长度
        prompt_len = int(model_inputs["attention_mask"][0].sum().item())
        prompt_lengths.append(prompt_len)

        # 真正至少需要的总长度
        required_len = prompt_len + target_len
        required_lengths.append(required_len)

        # ---------- 3) 视觉 token 数 ----------
        vision_len = None
        if "image_grid_thw" in model_inputs:
            grid = model_inputs["image_grid_thw"][0].tolist()  # [t, h, w]
            vision_len = int(grid[0] * grid[1] * grid[2])
        elif "pixel_attention_mask" in model_inputs:
            vision_len = int(model_inputs["pixel_attention_mask"][0].sum().item())

        vision_token_lengths.append(vision_len)

        # 只打印前 20 条，避免刷屏
        if i < 20:
            print(
                f"sample {i}: "
                f"target_len={target_len}, "
                f"prompt_len={prompt_len}, "
                f"required_len={required_len}, "
                f"vision_tokens={vision_len}"
            )

    # ---------- 汇总统计 ----------
    def pct(arr, p):
        arr_sorted = sorted(arr)
        idx = min(int(p * len(arr_sorted)), len(arr_sorted) - 1)
        return arr_sorted[idx]

    print("\n===== target_len stats =====")
    print("max =", max(text_lengths))
    print("avg =", sum(text_lengths) / len(text_lengths))
    print("p95 =", pct(text_lengths, 0.95))
    print("p99 =", pct(text_lengths, 0.99))

    print("\n===== prompt_len stats =====")
    print("max =", max(prompt_lengths))
    print("avg =", sum(prompt_lengths) / len(prompt_lengths))
    print("p95 =", pct(prompt_lengths, 0.95))
    print("p99 =", pct(prompt_lengths, 0.99))

    print("\n===== required_len = prompt_len + target_len =====")
    print("max =", max(required_lengths))
    print("avg =", sum(required_lengths) / len(required_lengths))
    print("p95 =", pct(required_lengths, 0.95))
    print("p99 =", pct(required_lengths, 0.99))

    valid_vision = [x for x in vision_token_lengths if x is not None]
    if valid_vision:
        print("\n===== vision_tokens stats =====")
        print("max =", max(valid_vision))
        print("avg =", sum(valid_vision) / len(valid_vision))
        print("p95 =", pct(valid_vision, 0.95))
        print("p99 =", pct(valid_vision, 0.99))

    # 你原来的测试集划分保留
    _, dataset_test = model3508b.data_divide(subset, test_size=0.2, seed=CONFIG['seed'])
    dataset_gen_test = model3508b.data_divide_gen(subset, test_size=0.2, seed=CONFIG['seed'])


# In[4]:


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    optimizer = torch.optim.AdamW(
        (p for p in model3508b.model.parameters() if p.requires_grad),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )
    model3508b.train_run(
        dataset=subset,
        optimizer=optimizer,
        num_epochs=CONFIG["training"]["num_epochs"]["test1"],
    )
        


# In[4]:


if __name__ == "__main__":
    ## eval check
    if "model3508b" in globals() and "dataset_test" in globals():
        _metrics_check = model3508b.eval_run(dataset_test)
        print(_metrics_check)
    else:
        print("Run the setup cell first.")


# In[5]:


if __name__ == "__main__":
    ## gen eval check
    if "model3508b" in globals() and "dataset_gen_test" in globals():
        gen_metrics_check = model3508b.gen_eval_run(dataset_gen_test, max_new_tokens=8)
        print(gen_metrics_check)
    else:
        print("Run the setup cell first.")


# In[1]:




