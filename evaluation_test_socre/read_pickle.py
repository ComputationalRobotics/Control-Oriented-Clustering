import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import argparse
import wandb
import os
import yaml
import shutil

# other files
from utils import *
from pusht_env import *
from models import *
from eval_baseline import eval_baseline
import pickle
def main():

    part = 1
    all_result = []

    file = open(f'score_result_part{part}.pickle', 'rb')
    # dump information to that file
    all_result.extend(pickle.load(file))
    # close the file
    file.close()
    
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()