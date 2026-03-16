# Qwen3.5 Vision: Fine-Tuning Workflow in PyTorch

This project focuses on **multimodal fine-tuning practice** for a Qwen3.5-style vision-language model in PyTorch.  
The goal is to understand the full training and evaluation pipeline (data → model adaptation → metrics → analysis), rather than only running one inference command.

---

# Why This Project

The main objective is to learn practical fine-tuning behavior: parameter updates, model adaptation, and reproducible experiment tracking.  
The current implementation uses a pretrained VL backbone with **parameter-efficient adaptation (LoRA)** and multimodal preprocessing.

The pipeline includes:

- Vision-language input construction (image tokens + prompt + target alignment)
- Training/evaluation pipeline with reusable config and checkpoint resume
- LoRA-based selective training for efficient adaptation
- Metric logging and attention visualization for analysis

Fine-tuning data used in this project:

- Dataset: `unsloth/LaTeX_OCR`
- Format: image + LaTeX text target
- Strategy: parameter-efficient adaptation with LoRA

---

# Core Learning Outcomes

This project demonstrates several practical multimodal training skills:

- Built an **end-to-end multimodal training pipeline** in PyTorch notebooks
- Implemented **robust data transforms** for vision + text alignment
- Applied **LoRA-based fine-tuning** with resume-ready checkpoint flow
- Tracked experiment metrics and exported artifacts (JSON / CSV / PNG)
- Evaluated model quality on unseen samples
- Visualized attention behavior for model interpretation

---

# Model and Training Configuration

## Training Dynamics

The figure below shows the training loss across optimization steps during LoRA fine-tuning.

The loss decreases rapidly during the early training steps and gradually stabilizes afterward, indicating effective adaptation of the model to the OCR-style image-to-text task. The loss curve shown here is plotted from an early-stage training run using a subset of the dataset for visualization purposes.


![Loss Curve](/outputs/train_logs/4b/loss_curve.png)
![Accuracy Curve](/outputs/train_logs/4b/accuracy_curve.png)



Base model:

Qwen3.5-VL 4B

Fine-tuning method:

LoRA (Low-Rank Adaptation)

LoRA configuration:

rank = 8  
alpha = 16  
dropout = 0.05  

Training environment:

Framework: PyTorch Distributed Data Parallel (DDP)  
GPUs: 2 × NVIDIA V100 (16GB)  
Per-GPU batch size: 2  
Effective batch size: 4  
Gradient checkpointing: enabled  
use_cache: disabled automatically for checkpoint compatibility  

Experiments were conducted on a university high-performance computing cluster using two NVIDIA V100 GPUs with PyTorch Distributed Data Parallel (DDP).

---

# Dataset Configuration

Fine-tuning dataset:

Dataset: `unsloth/LaTeX_OCR`  
Dataset size: **68,686 samples**

To avoid loading the entire dataset into memory, the pipeline uses **chunk-based training**.

Chunk size: 8000 samples  
Epochs per chunk: 2  
Resume training: enabled  

This divides the dataset into:

8 full chunks (8000 samples each)  
1 final chunk (4686 samples)

Chunk-based training allows the model to process large datasets without exceeding memory limits on cluster GPUs.

---

# Input Representation

Images are resized and converted into vision tokens before being combined with the text prompt.

Configuration:

Image size: **40 × 250**  
Prompt token length: **384**  
Maximum text length: **384**

This ensures the model has enough capacity for both prompt tokens and generated LaTeX sequences.

---

# Token Statistics

Before training, a token-length inspection was performed to determine safe sequence length limits.

Observed statistics:

Maximum target text length: **241**  
Average target text length: **~70**  
95th percentile: **138**  
99th percentile: **165**  
Prompt token length: **85**  
Maximum combined length needed: **326**  
Vision tokens per image: **320**

Based on these results:

max_text_len = 384  
prompt_max_len = 384  

This provides a safe margin and prevents truncation during training.

---

# Runtime Performance

Observed runtime performance on the SCC cluster:

Step time: **1.68s – 1.85s**  
Typical step speed: **~1.75s per step**

GPU monitoring shows:

GPU utilization: **~65% – 70%**  
GPU memory usage: **~14.95GB / 16GB**

This indicates that the training process is **stable and close to GPU memory capacity** while maintaining active compute usage.

---

# Training Behavior

During training, the loss decreases rapidly and remains stable with normal batch fluctuations.

These fluctuations are expected because:

- batch size is relatively small
- sample difficulty varies
- multimodal OCR tasks produce noisier per-step loss

No instability such as **NaN loss or exploding gradients** was observed.

---

# Training Time Estimate

With the current configuration:

Dataset size: **68,686**  
Epochs per chunk: **2**  
Effective batch size: **4**

Estimated steps:

8 chunks × (8000 / 4) × 2 ≈ **32,000 steps**  
final chunk ≈ **2,344 steps**

Total ≈ **34,344 steps**

Estimated runtime:

~1.75s per step  
Total training time ≈ **16 – 18 hours**

Estimated GPU usage:

≈ **32 – 35 GPU-hours**

Because SCC GPU sessions are typically **7 hours**, training relies on:

**resume-based multi-session execution**

---

# Recommended Configuration

```
DATASET_SIZE = 68686
CHUNK_SIZE = 8000
MODEL_SIZE = "4b"
NUM_EPOCHS = 2
RESUME = True
```

Data configuration:

```
image_size_height = 40
image_size_width = 250
image_token_len = 384
max_text_len = 384
```

---

# Recommended Notebook Structure

Use the notebooks below in order:

1. `instruction.ipynb`  
   Project instructions and scope.

2. `01_setup.ipynb`  
   Set up dependencies, seed control, and central config.

3. `02_data.ipynb`  
   Load dataset, inspect samples, and save local dataset artifact.

4. `03_model.ipynb`  
   Define model wrapper, multimodal transforms, training/eval/generation helpers.

5. `04_train.ipynb`  
   Run fine-tuning jobs, save model/metrics, and support resume across sessions.

6. `04_train_ddp.ipynb`  
   Run fine-tuning jobs using Distributed Data Parallel (DDP), save model checkpoints and metrics, and support resume across training sessions.

7. `05_eval_attention.ipynb`  
   Run evaluation metrics and generate attention visualization artifacts.

8. `06_main.ipynb`  
   Sets training parameters (e.g., dataset size, chunk size, epochs) and launches chunk-based training by calling `04_train_ddp.py`.


---

# Suggested Project Layout

```
.
|-- README.md
|-- instruction.ipynb
|-- LICENSE.txt
|
|-- 01_setup.ipynb
|-- 02_data.ipynb
|-- 03_model.ipynb
|-- 04_train_ddp.ipynb
|-- 05_eval_attention.ipynb
|-- 06_main.ipynb
|
|-- setup_01.py
|-- data_02.py
|-- model_03.py
|-- train_04.py
|-- train_04_ddp.py
|
|-- checkpoints/
|-- outputs/
\-- vision_text.pt
```

# Dataset License Note

This project uses the Unsloth dataset for fine-tuning.

License: **Apache License 2.0**  
URL: https://www.apache.org/licenses/LICENSE-2.0  

Attribution:  
Copyright 2024  
Unsloth AI, Daniel Han-Chen, and Michael Han-Chen
