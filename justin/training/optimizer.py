from torch.optim import AdamW

from ..config import Config
from ..model import SystemOne
from ..model.lora import LoRALinear


def build_optimizer(model: SystemOne, config: Config) -> AdamW:
    """Optimize only trainable parameters; frozen encoders receive no optimizer state.

    LoRA adapters (backbone) get their own smaller learning rate and no weight
    decay, since adapting 1280-dim attention blocks at the head's LR diverges.
    """
    lora_ids = {
        id(p)
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for p in (*module.a.parameters(), *module.b.parameters())
    }
    head, adapters = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (adapters if id(parameter) in lora_ids else head).append(parameter)
    groups = [{"params": head, "lr": config.learning_rate, "weight_decay": config.weight_decay}]
    if adapters:
        groups.append({"params": adapters, "lr": config.lora_learning_rate, "weight_decay": 0.0})
    return AdamW(groups)
