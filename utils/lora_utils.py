import torch
import torch.nn as nn
import peft


@torch.no_grad()
def custom_init(model,
                mode : str='initA'):
    assert mode in ['initA', 'initB'], f'Undefined mode {mode}'
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            if mode == 'initA':
                nn.init.kaiming_normal_(module.lora_A['default'].weight, nonlinearity='linear')
                nn.init.zeros_(module.lora_B['default'].weight)
            else:
                nn.init.kaiming_normal_(module.lora_B['default'].weight, nonlinearity='linear')
                nn.init.zeros_(module.lora_A['default'].weight)
