# Qwen3.5 Vision: From-Scratch Fine-Tuning in PyTorch

This project builds and fine-tunes a **Qwen3.5-style multimodal Transformer** from first principles using PyTorch.  
The goal is not just to run a pretrained model, but to understand and implement the full learning pipeline for image-text generation.

## Why This Project

Most tutorials treat multimodal models as black boxes. This project focuses on implementation-level understanding by designing the core components directly:

- A **vision encoder** for image feature extraction
- A **text Transformer decoder** with explicit Q/K/V attention computation
- **Cross-attention blocks** connecting visual tokens to text tokens
- A **token prediction head** for autoregressive response generation

Fine-tuning is done on the Unsloth dataset:

- Dataset: `unsloth/qwen3_5_vision`
- Format: image + instruction prompt + target response
- Strategy: **selective unfreezing** (freeze stable pretrained blocks, update cross-attention/output layers)

## Core Learning Outcomes

- Implemented a multimodal Transformer architecture in raw PyTorch
- Built a full fine-tuning loop: dataloading, optimizer, scheduler, loss, and validation
- Managed multimodal preprocessing (image transforms + text tokenization/alignment)
- Applied parameter-efficient fine-tuning by controlling trainable modules
- Evaluated model quality on unseen samples and visualized attention behavior

## Recommended Notebook Structure

Use the notebooks below in order:

1. `notebooks/01_environment_and_config.ipynb`  
	Set up dependencies, seeds, configs, and experiment paths.

2. `notebooks/02_data_pipeline_unsloth_qwen35v.ipynb`  
	Load `unsloth/qwen3_5_vision`, preprocess images/prompts/targets, create PyTorch dataset + dataloaders.

3. `notebooks/03_model_architecture_from_scratch.ipynb`  
	Implement vision encoder, text decoder blocks, explicit self-attention, cross-attention, and LM head.

4. `notebooks/04_finetuning_selective_layers.ipynb`  
	Load initial weights, freeze selected modules, train cross-attention + output layers, track metrics.

5. `notebooks/05_evaluation_and_attention_visualization.ipynb`  
	Run validation/inference on unseen examples, compute metrics, and visualize attention maps.

6. `notebooks/06_inference_demo.ipynb`  
	Lightweight interactive demo for single image + prompt prediction.

## Suggested Project Layout

```text
.
|-- README.md
|-- notebooks/
|   |-- 01_environment_and_config.ipynb
|   |-- 02_data_pipeline_unsloth_qwen35v.ipynb
|   |-- 03_model_architecture_from_scratch.ipynb
|   |-- 04_finetuning_selective_layers.ipynb
|   |-- 05_evaluation_and_attention_visualization.ipynb
|   \-- 06_inference_demo.ipynb
|-- src/
|   |-- models/
|   |-- data/
|   |-- train/
|   \-- utils/
|-- outputs/
|   |-- checkpoints/
|   |-- logs/
|   \-- figures/
\-- requirements.txt
```

## Resume-Ready Summary

Implemented and fine-tuned a Qwen3.5-style vision-language Transformer from scratch in PyTorch using the Unsloth `unsloth/qwen3_5_vision` dataset; engineered explicit self-attention and cross-attention modules, applied selective layer unfreezing for efficient adaptation, and evaluated performance with qualitative outputs and attention visualization.

## Interview Talking Points

- Why cross-attention is the key multimodal bridge
- What changed when freezing vs unfreezing deeper blocks
- Tradeoffs between full fine-tuning and selective adaptation
- Failure cases observed during inference and how they were mitigated
