#!/usr/bin/env python
# coding: utf-8

# # 01 - Environment and Config
# Set up packages, reproducibility, and training configuration.

# In[1]:


# Core imports
import os
import random
import numpy as np
import torch
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


# Central config dictionary
CONFIG = {
    "project_name": "qwen35v_from_scratch",
    "seed": 42,
    "device": device,
    "dtype": torch.float16 if device == "cuda" else torch.float32,

    "models": {
        "0.8b": "Qwen/Qwen3.5-0.8B",
        "4b": "Qwen/Qwen3.5-4B",
        "9b": "Qwen/Qwen3.5-9B",
    },

    "hf": {
        "token_env": "HF_TOKEN",
        "trust_remote_code": True,
    },

    "data": {
        "image_size_height": 40,
        "image_size_width": 250,
        "image_token_len": 384,
        "max_text_len": 384,
    },
    ## main.ipynb
    "training": {
        "batch_size": 2,
        "num_workers": 2,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,

        # 兼容旧代码
        "num_epochs": {
            "test1": 3,
            "test2": 10,
            "test3": 30
        }
    },
    ## just for compatiable.
    "train_session": {
        "dataset_size": 2,
        "model_size": "0.8b",
        "resume_training": True,
        "state_dir": "outputs/train_logs",
        "num_epochs": 10,
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
        },
    },

    "optimization": {
        "gradient_accumulation_steps": 1,
        "warmup_steps": 500,
    },
}


# In[3]:


if __name__ == "__main__":
    print(CONFIG)
    token_name = CONFIG["hf"]["token_env"]
    print(f"HF token env ({token_name}) exists:", bool(os.getenv(token_name)))


# In[1]:




