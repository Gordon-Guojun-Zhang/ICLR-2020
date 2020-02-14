import torch
import os
import numpy as np

def init_seed(seed=12):
##################################### cuda/GPU ########################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

# fixing the seed makes the convergence bad, no idea why
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return device
