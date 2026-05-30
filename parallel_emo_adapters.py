import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

class parallel_emo_adapters(nn.Module):
    def __init__(self,base_model,num_emotions=5,r=8,alpha=16):
        super().__init__()
        self.base_model = base_model
        self.num_emotions = num_emotions
        self.r = r  
        self.scaling=alpha/r

    #@staticmethod
    def create_adapter(base_model,r,alpha):
        adapter_config=LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q", "v", "k"], 
            lora_dropout=0.05,
            task_type="seq2seq_lm"
        )
        adapter=get_peft_model(base_model,adapter_config)
        return adapter
    
