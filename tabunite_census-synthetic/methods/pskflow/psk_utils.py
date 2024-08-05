import torch
import math

def convert_psk(tensor, categories):
    device = tensor.device
    tensor = tensor.int()
    cat_spacings = 2 * math.pi / torch.tensor(categories).to(device)
    
    phases = tensor * cat_spacings
    psk_enc = torch.cat((torch.sin(phases), torch.cos(phases)), dim=-1)
    return psk_enc

def psk_to_cat(psk_tensor, categories):
    device = psk_tensor.device
    cat_len = categories.shape[0]
    cat_spacings = 2 * math.pi / torch.tensor(categories).to(device)
    
    sin_phases = psk_tensor[:, :cat_len]
    cos_phases = psk_tensor[:, cat_len:2*cat_len]
    
    phases = torch.atan2(sin_phases, cos_phases)
    
    # Normalize phases to [0, 2pi] THIS IS VERY IMPORTANT BECAUSE ATAN COULD RESULT IN NEGATIVE VALUES
    phases = (phases + 2 * math.pi) % (2 * math.pi)
    
    feature_cat = torch.round(phases / cat_spacings).long()
    for i, cat in enumerate(categories):
        feature_cat[:, i] = torch.where(feature_cat[:, i] >= cat, 0, feature_cat[:, i])
    
    return feature_cat