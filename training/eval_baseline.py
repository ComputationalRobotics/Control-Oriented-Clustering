import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from skvideo.io import vwrite
import os
import argparse
import json

# dp defined utils
from utils import *
from pusht_env import *
from models import *


def eval_baseline(config, dataset_name, num_datasets, combined_stats, models_save_dir):

    # Your training code here
    # For demonstration, we'll just print the values
    num_diffusion_iters = config['num_diffusion_iters']
    num_tests = config['num_tests']
    num_train_demos = config['num_train_demos']
    num_vis_demos = config['num_vis_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    output_dir = config['output_dir']
    resize_scale = config["resize_scale"]



    resize_scale = 96

    if num_vis_demos > num_tests:
        num_vis_demos = num_tests 

    output_dir_good_vis = os.path.join(output_dir, "good_vis")

    print("Training parameters:")
    print(f"num_diffusion_iters: {num_diffusion_iters}")
    print(f"num_tests: {num_tests}")
    print(f"num_train_demos: {num_train_demos}")
    print(f"num_vis_demos: {num_vis_demos}")
    print(f"pred_horizon: {pred_horizon}")
    print(f"obs_horizon: {obs_horizon}")
    print(f"action_horizon: {action_horizon}")


    ##################### Instantiating Model and EMA #####################


    nets = nn.ModuleDict({})

    vision_encoder = get_resnet()
    vision_encoder = replace_bn_with_gn(vision_encoder)


    nets['vision_encoder'] = vision_encoder


    vision_feature_dim = 512

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

    nets = nets.to(device)

    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    ##################### LOADING Model and EMA #####################
        
    for model_name, model in nets.items():
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)

    ema_nets = nets
    ema_path = os.path.join(models_save_dir, f"ema_nets.pth")
    model_state_dict = torch.load(ema_path)
    ema.load_state_dict(model_state_dict)
    ema.copy_to(ema_nets.parameters())

    print("All models have been loaded successfully.")


    ##################### Prepare Inference #####################

    # Standard Gym Env (0.21.0 API)

    ##################### Start Inference #####################

    # (num_domains, num_tests)
    scores = [] 
    json_dict = dict()
    for domain_j in range(num_datasets):
        env_j_scores = []
        env_seed = 100000   # first test seed

        with open("./domains_yaml/{}.yml".format(dataset_name[domain_j]), 'r') as stream:
            data_loaded = yaml.safe_load(stream)        
        env_id = data_loaded["domain_id"]

        json_dict["domain_{}".format(env_id)] = []

        print("\nEval Diff Policy on Domain #{}:".format(env_id))

        for test_index in range(num_tests):
            noise_scheduler = create_injected_noise(num_diffusion_iters)

            # limit enviornment interaction to 300 steps before termination
            max_steps = config["max_steps"]
            env = PushTImageEnv(domain_filename=dataset_name[domain_j], resize_scale=resize_scale, pretrained=False)
            # use a seed >600 to avoid initial states seen in the training dataset
            env.seed(env_seed)
            # get first observation
            obs, info = env.reset()
            # keep a queue of last 2 steps of observations
            obs_deque = collections.deque(
                [obs] * obs_horizon, maxlen=obs_horizon)
            # save visualization and rewards
            imgs = [env.render(mode='rgb_array')]
            rewards = list()
            done = False
            step_idx = 0

            tqdm._instances.clear()
            with tqdm(total=max_steps, desc="Eval Trial #{}".format(test_index)) as pbar:
                while not done:
                    B = 1
                    # stack the last obs_horizon number of observations
                    images = np.stack([x['image'] for x in obs_deque])
                    agent_poses = np.stack([x['agent_pos'] for x in obs_deque])

                    # normalize observation
                    nagent_poses = normalize_data(agent_poses, stats=combined_stats[domain_j]['agent_pos'])
                    
                    # device transfer
                    nimages = torch.from_numpy(images).to(device, dtype=torch.float32)
                    # (2,3,96,96)
                    nagent_poses = torch.from_numpy(nagent_poses).to(device, dtype=torch.float32)
                    # (2,2)                

                    # infer action
                    with torch.no_grad():
                        # get image features
                        #image_features = ema_nets["vision_encoder"](nimages)

                        # encoder vision features
                        image_features = ema_nets["vision_encoder"](nimages)


                        image_features = image_features.squeeze()
                        # concat with low-dim observations
                        obs_features = torch.cat([image_features, nagent_poses], dim=-1)

                        # reshape observation to (B,obs_horizon*obs_dim)
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                        # initialize action from Guassian noise
                        noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
                        naction = noisy_action

                        # init scheduler
                        noise_scheduler.set_timesteps(num_diffusion_iters)

                        for k in noise_scheduler.timesteps:
                            # predict noise
                            noise_pred = ema_nets["invariant"](
                                sample=naction,
                                timestep=k,
                                global_cond=obs_cond
                            )                   

                            # inverse diffusion step (remove noise)
                            naction = noise_scheduler.step(
                                model_output=noise_pred,
                                timestep=k,
                                sample=naction
                            ).prev_sample

                    # unnormalize action
                    naction = naction.detach().to('cpu').numpy()
                    # (B, pred_horizon, action_dim)
                    naction = naction[0]
                    action_pred = unnormalize_data(naction, stats=combined_stats[domain_j]['action'])

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end,:]
                    # (action_horizon, action_dim)

                    # execute action_horizon number of steps
                    # without replanning
                    for i in range(len(action)):
                        # stepping env
                        obs, reward, done, _, info = env.step(action[i])
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        rewards.append(reward)
                        imgs.append(env.render(mode='rgb_array'))

                        # update progress bar
                        step_idx += 1
                        pbar.update(1)
                        pbar.set_postfix({"current": reward, "max": max(rewards)})
                        if step_idx > max_steps:
                            done = True
                        if done:
                            break
            
            env_seed += 1
            env_j_scores.append(max(rewards))


        print("Single DP on Domain #{} Avg Score: {}".format(env_id, np.mean(env_j_scores)))

    ############################ Save Result  ############################ 
        scores.append(env_j_scores)    

    with open(os.path.join(output_dir_good_vis, 'result.json'), 'w') as fp:
        json.dump(json_dict, fp)

    with open(os.path.join(output_dir_good_vis, 'all_result.json'), 'w') as fp_all:
        json.dump(scores, fp_all)
    print("Eval done!")
    return scores

