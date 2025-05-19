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

def main():
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/baseline.yml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    
    num_train_demos = config['num_train_demos'] 
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    output_dir = config['output_dir']
    models_save_dir = config['models_save_dir']
    dataset_path_dir = config['dataset_path_dir']
    adapt_dataset_path_dir = config['adapt_dataset_path_dir']
    resize_scale = config["resize_scale"]

    if config["use_pretrained"]:
        print("Load and freeze pretrained ResNet18")
        resize_scale = 224
    else:
        print("Unfreeze and update ResNet18")

    num_trained_datasets = 0
    for entry in sorted(os.listdir(dataset_path_dir)):
        if entry[-5:] == '.zarr':
            num_trained_datasets += 1

    if config["adapt"]:
        dataset_path_dir = adapt_dataset_path_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_stats = []
    num_datasets = 0
    dataset_name = {} # mapping for domain filename

    ##################### Instantiating Dataset #####################

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
            pretrained=config["use_pretrained"]
        )
        num_datasets += 1
        # save training data statistics (min, max) for each dim
        stats = dataset.stats
        combined_stats.append(stats)

    eval_baseline(config, dataset_name, num_datasets, combined_stats, models_save_dir)

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
    verbose = config['verbose']
    output_dir = config['output_dir']
    resize_scale = config["resize_scale"]

    if config["use_pretrained"]:
        print("Load and freeze pretrained ResNet18")
        resize_scale = 224

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
    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    nets = nn.ModuleDict({})

    vision_encoder = get_resnet()
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets['vision_encoder'] = vision_encoder

    # if config["use_mlp"]:
    #     nets["invariant_fc"] = DropoutMLP(input_dim=vision_feature_dim, 
    #                                         hidden_dim=1024,
    #                                         output_dim=vision_feature_dim,
    #                                         num_layers=2
    #                                         )  

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
            if verbose:
                # 0. create env object
                env = PushTImageEnv(domain_filename=dataset_name[domain_j], resize_scale=resize_scale, pretrained=config["use_pretrained"])
                # 1. seed env for initial state.
                # Seed 0-600 are used for the demonstration dataset.
                env.seed(1000)
                # 2. must reset before use
                obs, info = env.reset()
                # 3. 2D positional action space [0,512]
                action = env.action_space.sample()
                # 4. Standard gym step method
                obs, reward, terminated, truncated, info = env.step(action)

                # prints and explains each dimension of the observation and action vectors
                with np.printoptions(precision=4, suppress=True, threshold=5):
                    print("obs['image'].shape: {}, {}, [{},{}]".format(obs['image'].shape, obs['image'].dtype, np.min(obs['image']), np.max(obs['image'])))
                    print("obs['agent_pos'].shape: {}, {}, [{},{}]".format(obs['agent_pos'].shape, obs['agent_pos'].dtype, np.min(obs['agent_pos']), np.max(obs['agent_pos'])))
                    print("action.shape: {}, {}, [{},{}]".format(action.shape, action.dtype, np.min(action), np.max(action)))

            # limit enviornment interaction to 300 steps before termination
            max_steps = config["max_steps"]
            env = PushTImageEnv(domain_filename=dataset_name[domain_j], resize_scale=resize_scale, pretrained=config["use_pretrained"])
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
                        image_features = ema_nets["vision_encoder"](nimages)
                        # if config["use_mlp"]:
                        #     image_features = nets["invariant_fc"](image_features)
                        # (2,512)

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
            # save the visualization of the first few demos
            if test_index < num_vis_demos:
                vwrite(os.path.join(output_dir, "baseline_single_dp_on_domain_{}_test_{}.mp4".format(env_id, test_index)), imgs)

            if max(rewards) > 0.8:
                vwrite(os.path.join(output_dir_good_vis, "baseline_single_dp_on_domain_{}_test_{}.mp4".format(env_id, test_index)), imgs)
                cur_dict = {"trial_id": test_index, "score": max(rewards)}
                json_dict["domain_{}".format(env_id)].append(cur_dict)

        print("Single DP on Domain #{} Avg Score: {}".format(env_id, np.mean(env_j_scores)))

    ############################ Save Result  ############################ 
        scores.append(env_j_scores)    

    with open(os.path.join(output_dir_good_vis, 'result.json'), 'w') as fp:
        json.dump(json_dict, fp)

    print("Eval done!")
    return scores

if __name__ == "__main__":
    main()
