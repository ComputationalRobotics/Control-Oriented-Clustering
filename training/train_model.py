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

def main():
    np.random.seed(2)
    torch.manual_seed(2)
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/train_config.yml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    
    num_epochs = config['num_epochs']
    num_diffusion_iters = config['num_diffusion_iters']
    num_tests = config['num_tests']
    num_vis_demos = config['num_vis_demos']
    num_train_demos = config['num_train_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    lr = config['lr']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    dataset_path_dir = config['dataset_path_dir']
    output_dir = config['output_dir']
    models_save_dir = config['models_save_dir']
    display_name = config['display_name']
    resize_scale = config["resize_scale"]

    if display_name == "default":
        display_name = None
    if config["wandb"]:
        wandb.init(
            project="train_model",
            config=config,
            name=display_name
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

    print("Use default AdamW as optimizer.")


    output_dir_good_vis = os.path.join(output_dir, "good_vis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir_good_vis)

    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

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


    nets = nn.ModuleDict({})
    noise_schedulers = {}


    vision_encoder = get_resnet()
    vision_feature_dim = 512
    vision_encoder = replace_bn_with_gn(vision_encoder)

    nets['vision_encoder'] = vision_encoder

    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    invariant = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    nets['invariant'] = invariant 
    
    noise_schedulers["single"] = create_injected_noise(num_diffusion_iters)        

    nets = nets.to(device)


    # Exponential Moving Average accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parameters are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=lr, weight_decay=weight_decay)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=len(dataloader) * 3000
    )


    with tqdm(range(1, num_epochs+1), desc='Epoch') as tglobal:
        # unique_ids = torch.arange(num_datasets).cpu()
        # epoch loop
        for epoch_idx in tglobal:
            if config['wandb']:
                wandb.log({'epoch': epoch_idx})    
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    if config["wandb"]:
                        wandb.log({'learning_rate:': lr_scheduler.get_last_lr()[0]})
                    
                    # device transfer
                    # data normalized in dataset
                    nimage = nbatch['image'][:,:obs_horizon].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                    naction = nbatch['action'].to(device)
                    B = nagent_pos.shape[0]

                    # encoder vision features
                    image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))

                    image_features = image_features.reshape(*nimage.shape[:2],-1)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noises to add to actions
                    noise= torch.randn(naction.shape, device=device)
                    
                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                            0, noise_schedulers["single"].config.num_train_timesteps,
                            (B,), device=device).long()
                    
                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_schedulers["single"].add_noise(
                        naction, noise, timesteps)
                    
                    # predict the noise residual
                    noise_pred = nets["invariant"](noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    if config['wandb']:
                        wandb.log({'loss': loss_cpu, 'epoch': epoch_idx})
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # save and eval upon request
            if (epoch_idx <= 75 and epoch_idx % 10 == 0) or (epoch_idx > 75 and epoch_idx % 20 == 0) or (epoch_idx in [num_epochs]) or (epoch_idx == 1):
                # create new checkpoint
                checkpoint_dir = '{}/checkpoint_epoch_{}'.format(models_save_dir, epoch_idx)

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                save(ema, nets, checkpoint_dir)
                if  (epoch_idx in [num_epochs]):
                    scores = eval_baseline(config, dataset_name, num_datasets, combined_stats, checkpoint_dir)
                    # (num_domains, num_tests)
                    
                    if config["wandb"]:
                        for domain_j, domain_j_scores in enumerate(scores):

                            with open("./domains_yaml/{}.yml".format(dataset_name[domain_j]), 'r') as stream:
                                data_loaded = yaml.safe_load(stream)
                            env_id = data_loaded["domain_id"]

                            wandb.log({"baseline_single_dp_on_domain_{}_avg_eval_score".format(env_id): np.mean(domain_j_scores), 'epoch': epoch_idx})

                        wandb.log({"baseline_single_dp_on_all_domains_avg_eval_score": np.mean(scores), 'epoch': epoch_idx})
                        for i in range(10):
                            threshold = 0.1*i
                            count = (np.array(scores)>threshold).sum()
                            wandb.log({"num_tests_threshold_{:.1f}".format(threshold): count, 'epoch': epoch_idx})
                
if __name__ == "__main__":
    main()