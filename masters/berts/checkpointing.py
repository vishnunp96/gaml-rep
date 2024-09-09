import os
import torch
from masters.berts.helpers import fprint

def save_checkpoint(model, base_model_name, optimizer, scheduler, batch, epoch, loss,
                    model_save_path, prev_model_save_path):
    if os.path.exists(model_save_path):
        if os.path.exists(prev_model_save_path):
            os.remove(prev_model_save_path)
        os.rename(model_save_path, prev_model_save_path)
        fprint(f"Previous model moved to: {prev_model_save_path}", 0)
    torch.save({
        'batch': batch,
        'epoch': epoch,
        'base_model_name': base_model_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, model_save_path)
    fprint(f"Checkpoint saved: {model_save_path}", 0)


def load_checkpoint(model_save_path, model, optimizer=None, scheduler=None):
    if torch.cuda.is_available():
        print("Loading model on GPU")
        checkpoint = torch.load(model_save_path)
    else:
        checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
    model.load_state_dict(remove_module_prefix(checkpoint['model_state_dict']))
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    batch = checkpoint['batch']
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    base_model_name = checkpoint.get('base_model_name', "bert-base-cased")
    print(f"Checkpoint loaded: {model_save_path}")
    return batch, epoch, loss, base_model_name

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state_dict keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict