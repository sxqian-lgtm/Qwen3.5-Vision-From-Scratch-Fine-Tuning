#!/usr/bin/env python
# coding: utf-8

# # 02 - Data Pipeline
# Load and preprocess the `unsloth/qwen3_5_vision` dataset into model-ready batches.

# In[ ]:


from setup_01 import CONFIG, set_seed
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoProcessor
import numpy as np
import torch


# In[ ]:


def data_install():
    dataset = load_dataset("unsloth/LaTeX_OCR", split="train")
    return dataset

def example(dataset):
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(dataset))
    sample = dataset[idx]
    print(sample['image'])
    print(sample['text'])
    # Expected fields: image, text.
    return sample

def data_save(dataset, filename="vision_text.pt"):
    torch.save(dataset, filename)
    print(f"Dataset saved to {filename}")

def data_load(filename="vision_text.pt"):
    dataset = torch.load(filename, weights_only=False)
    return dataset



# In[ ]:


if __name__ == "__main__":
    set_seed(CONFIG['seed'])
    dataset = data_install()
    example1 = example(dataset)
    data_save(dataset, "vision_text.pt")


# In[ ]:




