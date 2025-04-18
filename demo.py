import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        dtype = original_layer.weight.dtype
        #原始权重
        self.original_weight = original_layer.weight
        self.original_bias = original_layer.bias
        
        #冻结初始权重
        self.original_weight.requires_grad_(False)
        if self.original_bias is not None:
            self.original_bias.requires_grad_(False)
        
        #设置LoRA参数
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.Lora_A = nn.Parameter(torch.randn(in_features, rank, dtype=dtype) * 0.02)
        self.Lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype))
        
    def forward(self, x):
        #原始层的输出
        original_out = F.linear (x, self.original_weight, self.original_bias)
        
        #LoRA层的输出
        lora_out = torch.matmul(torch.matmul (x, self.Lora_A), self.Lora_B) * (self.alpha / self.rank)
        return lora_out + original_out
    
def apply_lora (model, target_modules, rank, alpha):
    
    for name, module in model.named_children ():
        #如果不是叶子节点，则继续递归。
        if len(list(module.children())) > 0:
            apply_lora (module, target_modules, rank, alpha)
        #如果是叶子节点，则进行替换。
        if isinstance (module, nn.Linear) and name in target_modules:
            new_layer = LoRALayer (module, rank, alpha)
            setattr (model, name, new_layer)
    return model

#加载权重。
def load_lora_weight (model, lora_weights = None):
    for name, param in model.named_parameters ():
        if 'Lora_A' in name or 'Lora_B' in name:
            if lora_weights is None:
                param.data.zero_()
            elif name in lora_weights:
                param.data.copy_(lora_weights[name])
            else :
                print(f"Warning: {name} not found in loaded lora_weights.")
    
class loraConfig :
    rank = 8
    alpha = 16
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj']        
        