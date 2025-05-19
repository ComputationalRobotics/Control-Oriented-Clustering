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

    seed_lst = [2]
    ck_epoch_lst = [1, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]


    all_result = [] #to save result
    for seed in seed_lst:
        for ck_epoch in ck_epoch_lst:
            dataset_name = {18: 'domain18'}
            checkpoint_dir = f"/home/jordan/Han/feature_analysis_code_for_submission/training/train_model_folder/checkpoint_epoch_{ck_epoch}"

            parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
            parser.add_argument('--config', type=str, default='./configs/test_domain_18_model.yml', help='Path to the configuration YAML file.')
            args = parser.parse_args()
            with open(args.config, 'r') as file:
                config = yaml.safe_load(file)
                
            #|o|o|                             observations: 2
            #| |a|a|a|a|a|a|a|a|               actions executed: 8
            #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            
            num_epochs = config['num_epochs']
            num_diffusion_iters = config['num_diffusion_iters']
            num_tests = config['num_tests']
            num_train_demos = config['num_train_demos']
            num_vis_demos = config['num_vis_demos']
            pred_horizon = config['pred_horizon']
            obs_horizon = config['obs_horizon']
            action_horizon = config['action_horizon']
            batch_size = config['batch_size']
            dataset_path_dir = config['dataset_path_dir']
            display_name = config['display_name']
            resize_scale = config["resize_scale"]

            display_name = display_name + f"_seed{seed}_ckepoch{ck_epoch}"


            if display_name == "default":
                display_name = None
            if config["wandb"]:                wandb.init(
                    project="test_model",
                    config=config,
                    name=display_name,
                    reinit=True
                )
            else:
                print("warning: wandb flag set to False")

            print("Training parameters:")
            print(f"num_epochs: {num_epochs}")
            print(f"num_diffusion_iters: {num_diffusion_iters}")
            print(f"num_tests: {num_tests}")
            print(f"num_train_demos: {num_train_demos}")
            print(f"num_vis_demos: {num_vis_demos}")
            print(f"pred_horizon: {pred_horizon}")
            print(f"obs_horizon: {obs_horizon}")
            print(f"action_horizon: {action_horizon}")


            print("\nBaseline Mode: Train Single Diffusion Policy")

           
            resize_scale = 96

            if num_vis_demos > num_tests:
                num_vis_demos = num_tests

            dataset_list = []
            combined_stats = []
            num_datasets = 0
            dataset_name = {} # mapping for domain filename

            for entry in sorted(os.listdir(dataset_path_dir)):
                if not (entry[-5:] == '.zarr'):
                    continue
                full_path = os.path.join(dataset_path_dir, entry)

                domain_filename = entry.split(".")[0]
                dataset_name[num_datasets] = domain_filename        

                # create dataset from file
                dataset = PushTImageDataset(
                    dataset_path=full_path,
                    pred_horizon=pred_horizon,
                    obs_horizon=obs_horizon,
                    action_horizon=action_horizon,
                    id = num_datasets,
                    num_demos = num_train_demos,
                    resize_scale = resize_scale,
                    pretrained = False
                )
                num_datasets += 1
                # save training data statistics (min, max) for each dim
                stats = dataset.stats
                dataset_list.append(dataset)
                combined_stats.append(stats)

            combined_dataset = ConcatDataset(dataset_list)

            # create dataloader
            dataloader = torch.utils.data.DataLoader(
                combined_dataset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=True,
                # accelerate cpu-gpu transfer
                pin_memory=True,
                # don't kill worker process afte each epoch
                persistent_workers=True
            )
            

            scores = eval_baseline(config, dataset_name, num_datasets, combined_stats, checkpoint_dir)


            if config["wandb"]:
                for domain_j, domain_j_scores in enumerate(scores):

                    with open("./domains_yaml/{}.yml".format(dataset_name[domain_j]), 'r') as stream:
                        data_loaded = yaml.safe_load(stream)
                    env_id = data_loaded["domain_id"]

                    wandb.log({"baseline_single_dp_on_domain_{}_avg_eval_score".format(env_id): np.mean(domain_j_scores), 'epoch': 300})
                
                
                wandb.log({"baseline_single_dp_on_all_domains_avg_eval_score": np.mean(scores), 'epoch': 300})
                for i in range(10):
                    threshold = 0.1*i
                    count = (np.array(scores)>threshold).sum()
                    wandb.log({"num_tests_threshold_{:.1f}".format(threshold): count, 'epoch': 300})

            save_score_result = {seed: {ck_epoch: np.mean(scores)}}
            all_result.append(save_score_result)

    with open(f'score_result.pickle', 'wb') as handle:
        pickle.dump(all_result, handle)

if __name__ == "__main__":
    main()