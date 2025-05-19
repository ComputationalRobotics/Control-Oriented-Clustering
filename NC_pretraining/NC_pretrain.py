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
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/NC_pretrain.yml', help='Path to the configuration YAML file.')
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
    eval_epoch = config['eval_epoch']
    lr = config['lr']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    dataset_path_dir = config['dataset_path_dir']
    output_dir = config['output_dir']
    models_save_dir = config['models_save_dir']
    verbose = config['verbose']
    display_name = config['display_name']
    resize_scale = config["resize_scale"]

    if display_name == "default":
        display_name = None
    if config["wandb"]:
        # wandb.login(key="c816a85f1488f7f1df913c6f7dae063d173d27b3") 
        wandb.init(
            project="real_world_training_09_28",
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
    print(f"eval_epoch: {eval_epoch}")


    print("\nBaseline Mode: Train Single Diffusion Policy")

    if config["use_pretrained"]:
        print("Load and freeze pretrained ResNet18")
    else:
        print("Unfreeze and update ResNet18")
    resize_scale = 96
    """
    if config["use_mlp"]:
        print("Insert a MLP between ResNet18 and Unet")
    """
    print("Use default AdamW as optimizer.")

    """
    if config["adapt"]:
        print("Adapt Mode activated!\n")
        dataset_path_dir = adapt_dataset_path_dir
    else:
        print("Adapt Mode deactivated!\n")
    """
    output_dir_good_vis = os.path.join(output_dir, "good_vis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir_good_vis)

    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

    if num_vis_demos > num_tests:
        num_vis_demos = num_tests


    #start for caliberated dataset and dataloader
    dataset_list_caliberated = []
    num_datasets_caliberated = 0
    dataset_name_caliberated = {} # mapping for domain filename

    for entry in sorted(os.listdir(dataset_path_dir)):
        if not (entry[-5:] == '.zarr'):
            continue
        full_path = os.path.join(dataset_path_dir, entry)

        domain_filename = entry.split(".")[0]
        dataset_name_caliberated[num_datasets_caliberated] = domain_filename        
        dataset = RealPushTImageDatasetCaliberated(
            dataset_path=full_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            id = num_datasets_caliberated,
            num_demos = num_train_demos,
            resize_scale = resize_scale,
            pretrained = config["use_pretrained"]
        )
        num_datasets_caliberated += 1
        # save training data statistics (min, max) for each dim
        stats = dataset.stats
        dataset_list_caliberated.append(dataset)

    combined_dataset_caliberated = ConcatDataset(dataset_list_caliberated)

    # create dataloader
    dataloader_caliberated = torch.utils.data.DataLoader(
        combined_dataset_caliberated,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    #end for caliberated dataset and dataloader





    if verbose:
        # visualize data in batch
        batch = next(iter(dataloader))
        print("batch['image'].shape: {}, {}, [{},{}]".format(batch['image'].shape, batch['image'].dtype, torch.min(batch['image']), torch.max(batch['image'])))
        print("batch['agent_pos'].shape: {}, {}, [{},{}]".format(batch['agent_pos'].shape, batch['agent_pos'].dtype, torch.min(batch['agent_pos']), torch.max(batch['agent_pos'])))
        print("batch['action'].shape: {}, {}, [{},{}]".format(batch['action'].shape, batch['action'].dtype, torch.min(batch['action']), torch.max(batch['action'])))
        print("batch['id']: {}, [{},{}]".format(batch['id'].shape, torch.min(batch['id']), torch.max(batch['id'])))



    nets = nn.ModuleDict({})
    noise_schedulers = {}

    # add one dp trained on all domains
    if config["use_pretrained"]:
        vision_encoder = get_resnet(weights='IMAGENET1K_V1')
        vision_feature_dim = 512
    else:
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


    

    if config["use_pretrained"]:
        for param in nets["vision_encoder"].parameters():
            param.requires_grad = False
    
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
        num_training_steps=(2*len(dataloader_caliberated)) * 3000
    )


    # create new checkpoint
    checkpoint_dir = '{}/checkpoint_epoch_{}'.format(models_save_dir, 0)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save(ema, nets, checkpoint_dir)


    #for the caliberated dataset
    with tqdm(dataloader_caliberated, desc='Batch', leave=False) as tepoch:
        for nbatch in tepoch:


            # device transfer
            # data normalized in dataset
            nimage_all = nbatch['image'][:,:obs_horizon].to(device, dtype = torch.float32)
            nagent_pos_all = nbatch['agent_pos'][:,:obs_horizon].to(device, dtype = torch.float32)
            naction_all = nbatch['action'].to(device, dtype = torch.float32)
            action_intention_all = torch.squeeze(nbatch['action_intention']).numpy()


    with tqdm(range(1, num_epochs+1), desc='Epoch') as tglobal:
        # unique_ids = torch.arange(num_datasets).cpu()
        # epoch loop
        for epoch_idx in tglobal:
            if config['wandb']:
                wandb.log({'epoch': epoch_idx})    
            epoch_loss = list()
            # batch loop
            nets["vision_encoder"].train()
            nets["invariant"].eval()
            
            #for the caliberated dataset
            for repeat in range(1):
                    K = 8
                    index_for_class = []
                    for k in range(K):
                        indices = [i for i, x in enumerate(action_intention_all) if x == k]
                        np.random.shuffle(indices)
                        index_for_class.append(indices)
                        
                        #print(len(action_intention_all))
                        #print(f"class{k}")
                        #print(len(indices))
                        
                        
                    num_division_bin = 80
                    #start_index_lst = [[0, 0, 0], [0, 0, 0], [0, 2857, 5714]]
                    #end_index_lst = [[len(index_for_class[0]), len(index_for_class[0]), len(index_for_class[0])], [len(index_for_class[1]), len(index_for_class[1]), len(index_for_class[1])], [2857, 5714, 8571]]
                    for i in range(num_division_bin):
                        nimage = []
                        nagent_pos = []
                        naction = []
                        action_intention =[]
                        for k in range(K):
                            start_index = i*int(len(index_for_class[k])/num_division_bin)
                            end_index = min((i+1)*int(len(index_for_class[k])/num_division_bin), len(index_for_class[k]))
                            
                            #start_index = start_index_lst[k][i]
                            #end_index = end_index_lst[k][i]


                            nimage.append(nimage_all[index_for_class[k]][start_index:end_index])
                            nagent_pos.append(nagent_pos_all[index_for_class[k]][start_index:end_index])
                            naction.append(naction_all[index_for_class[k]][start_index:end_index])
                            action_intention.append(action_intention_all[index_for_class[k]][start_index:end_index])

                        nimage = torch.vstack(nimage)
                        nagent_pos = torch.vstack(nagent_pos)
                        naction = torch.vstack(naction)
                        action_intention = np.concatenate(action_intention)

                        B = nagent_pos.shape[0]

                        # encoder vision features
                        image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))

                        image_features = image_features.reshape(*nimage.shape[:2],-1)
                        # (B,obs_horizon, 23*23)

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

                        
                        feature_result = torch.flatten(image_features, start_dim=1)

                        action_intention_result = action_intention


                        feature = feature_result
                        global_mean = torch.mean(feature, axis = 0)

                        label_lists = action_intention_result
                        result_list = []
                        for i in range(K):
                            result_list.append([])
                        for i, label in enumerate(label_lists):
                            if label != 8:
                                result_list[label].append(feature[i])
                        
                        cluster_mean = []
                        for i in range(K):
                            result_list[i] = torch.stack(result_list[i])
                            cluster_mean.append(torch.mean(result_list[i], axis=0) - global_mean)


                        globally_centered_class_mean_result = torch.stack(cluster_mean)


                        globally_centered_features = []
                        for i in range(K):
                            globally_centered_features.append(result_list[i] - global_mean)

                    
                        sigma_c_square = []
                        for i in range(K):

                            here = torch.sum(torch.stack([torch.square(torch.linalg.norm(globally_centered_features[i][j] - globally_centered_class_mean_result[i], 2)) for j in range(globally_centered_features[i].shape[0])]))
                            here = here/(1.0*(globally_centered_features[i].shape[0]-1))
                            sigma_c_square.append(here)
                        
                        within_class_cov = []
                        for i in range(K):
                            for j in range(K):
                                if i < j:
                                    within_class_cov.append((sigma_c_square[i]+sigma_c_square[j])/(2.0 * torch.square(torch.linalg.norm(globally_centered_class_mean_result[i] - globally_centered_class_mean_result[j], 2))))

                        #print("nc1")
                        nc1 = torch.mean(torch.stack(within_class_cov))
                        #print(nc1)


                        #for nc2
                        length_lst = []
                        angle_lst = []
                        for i in range(K):
                            length = torch.linalg.norm(globally_centered_class_mean_result[i], 2)
                            length_lst.append(length)


                        for i in range(K):
                            for j in range(K):
                                if i < j:
                                    angle = torch.inner(globally_centered_class_mean_result[i]/torch.linalg.norm(globally_centered_class_mean_result[i], 2), globally_centered_class_mean_result[j]/torch.linalg.norm(globally_centered_class_mean_result[j], 2))
                                    angle_lst.append(angle)

                        length_lst = torch.stack(length_lst)
                        angle_lst = torch.stack(angle_lst)
                        nc2_equinorm = torch.std(length_lst)/torch.mean(length_lst)
                        nc2_equiangularity = torch.std(angle_lst)

                        # optimize

                        #add regularization to the loss
                        #want to change it to 
                        #total_loss = loss + 10 * nc2_equinorm + 50 * nc2_equiangularity + 0.1 * nc1
                        total_loss =  10 * nc2_equinorm + 10 * nc2_equiangularity + 0.1 * nc1
                        
                        total_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        
                        lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        ema.step(nets.parameters())




                        # logging   
                        total_loss_cpu = total_loss.item()
                        loss_cpu = loss.item()
                        nc1_cpu = nc1.item()
                        nc2_equinorm_cpu = nc2_equinorm.item()
                        nc2_equiangularity_cpu = nc2_equiangularity.item()

                        if config['wandb']:
                            wandb.log({'loss': loss_cpu, 'epoch': epoch_idx, 'nc1': nc1_cpu, 'nc2_norm': nc2_equinorm_cpu, 'nc2_equiangularity': nc2_equiangularity_cpu, 'total_loss': total_loss_cpu})
                        epoch_loss.append(total_loss_cpu)
                        tepoch.set_postfix(loss=total_loss_cpu)


            tglobal.set_postfix(loss=np.mean(epoch_loss))
            # save and eval upon request
            if (epoch_idx % 50 == 0):
            #if epoch_idx % 3 == 0:
                # remove previous checkpoint
                
                pre_checkpoint_dir = os.listdir(models_save_dir)

                # create new checkpoint
                checkpoint_dir = '{}/checkpoint_epoch_{}'.format(models_save_dir, epoch_idx)

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                save(ema, nets, checkpoint_dir)
                
               
                
if __name__ == "__main__":
    main()