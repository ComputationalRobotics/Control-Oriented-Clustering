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
import time
import yaml
from easydict import EasyDict

# dp defined utils
from utils import *
from models import *

# hardware setups
from franka_control.deoxys.deoxys import config_root
from franka_control.deoxys.deoxys.franka_interface import FrankaInterface
from franka_control.deoxys.deoxys_vision.camera.rs_interface import RSInterface
import pyrealsense2 as rs
from franka_control.deoxys.deoxys.utils import YamlConfig
from franka_control.deoxys.deoxys.experimental.motion_utils import position_only_gripper_move_to

def get_connected_devices_serial():
    serials = list()
    for d in rs.context().devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            serial = d.get_info(rs.camera_info.serial_number)
            product_line = d.get_info(rs.camera_info.product_line)
            if product_line == 'D400':
                # only works with D400 series
                serials.append(serial)
    serials = sorted(serials)
    return serials

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

    ##################### Instantiating Dataset #####################

    for entry in sorted(os.listdir(dataset_path_dir)):
        if not (entry[-5:] == '.zarr'):
            continue
        full_path = os.path.join(dataset_path_dir, entry)

        domain_filename = entry.split(".")[0]

        # create dataset from file
        dataset = RealPushTImageDataset(
            dataset_path=full_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            id = 0,
            num_demos = num_train_demos,
            resize_scale = resize_scale,
            pretrained=config["use_pretrained"]
        )
        # save training data statistics (min, max) for each dim
        stats = dataset.stats

    eval_real_world(config, stats, models_save_dir)

def eval_real_world(config, stats, models_save_dir):

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


    ##################### Instantiate Model and EMA #####################
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

    ##################### Load Model and EMA #####################
        
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


    ##################### Instantiate Hardware Inferfaces #####################

    # Franka Robot Arm Interface
    controller_type = "OSC_POSITION"
    controller_cfg_filename = "osc-position-controller.yml"
    controller_cfg = YamlConfig(os.path.join(config_root, controller_cfg_filename)).as_easydict()
    interface_cfg_filename = "pc_franka.yml"
    robot_interface = FrankaInterface(os.path.join(config_root, interface_cfg_filename), has_gripper=False, control_freq=30)

    # Intel Realsense Cameras
    serials = get_connected_devices_serial()
    img_w = 320
    img_h = 240
    camera_ids = range(len(serials))
    cr_interfaces = {}
    for camera_id in camera_ids:
        color_cfg = EasyDict(enabled=True, img_w=img_w, img_h=img_h, img_format=rs.format.bgr8, fps=30)
        depth_cfg = EasyDict(enabled=False, img_w=img_w, img_h=img_h, img_format=rs.format.z16, fps=30)
        pc_cfg = EasyDict(enabled=False)
        cr_interface = RSInterface(device_id=serials[camera_id], color_cfg=color_cfg, depth_cfg=depth_cfg, pc_cfg=pc_cfg)
        cr_interface.start()
        cr_interfaces[camera_id] = cr_interface
    
    time.sleep(2)

    # initialize pose to specific location
    initial_pos = np.array([[0.25],[0.38],[0.2]])
    position_only_gripper_move_to(robot_interface, initial_pos, num_steps=100, controller_cfg=controller_cfg)

    time.sleep(1)

    ##################### Start Inference #####################
    
    # (num_steps)
    scores = [] 
    noise_scheduler = create_injected_noise(num_diffusion_iters)
    # limit enviornment interaction to certain number of steps before termination
    max_steps = config["max_steps"]
    camera_frames = np.zeros((len(serials), 9999, img_h, img_w, 3), dtype=np.uint8)

    # IoU metric
    rewards = list()
    done = False
    crop_h_1 = 4
    crop_h_2 = 228
    crop_w_1 = 35
    crop_w_2 = 259

    if config["use_pretrained"]:
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(resize_scale),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(resize_scale),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # get first observation
    for camera_id in camera_ids:
        img_color = cr_interfaces[camera_id].get_color_img()
        camera_frames[camera_id][0] = img_color
    img_obs = transform(camera_frames[1][0][crop_h_1:crop_h_2, crop_w_1:crop_w_2, :])
    agent_pos = robot_interface.last_eef_pos_and_rotvec
    agent_pos_xy = agent_pos[:2]
    obs = {'image': img_obs,
           'agent_pos': agent_pos_xy}

    step_idx = 1
    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

    tqdm._instances.clear()
    with tqdm(total=max_steps, desc="Real-world Eval") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon number of observations
            images = np.stack([x['image'] for x in obs_deque])
            # (2,3,resize_scale,resize_scale)
            agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
            # (2,2)

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])
            
            # device transfer
            nimages = torch.from_numpy(images).to(device, dtype=torch.float32)
            # (2,3,resize_scale,resize_scale)
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
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                action_6dof = np.zeros(6)
                action_6dof[:2] = action[i]
                robot_interface.control(
                    controller_type=controller_type,
                    action=action_6dof,
                    controller_cfg=controller_cfg,
                )

                for camera_id in camera_ids:
                    img_color = cr_interfaces[camera_id].get_color_img()
                    camera_frames[camera_id][step_idx] = img_color
                img_obs = transform(camera_frames[1][step_idx][crop_h_1:crop_h_2, crop_w_1:crop_w_2, :])
                agent_pos = robot_interface.last_eef_pos_and_rotvec
                agent_pos_xy = agent_pos[:2]
                obs = {'image': img_obs,
                    'agent_pos': agent_pos_xy}

                # save observations
                obs_deque.append(obs)
                # and reward/vis
                # TODO: use SAM to get current position of block to compute IoU
                reward = compute_IoU()
                rewards.append(reward)

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix({"current": reward, "max": max(rewards)})
                if step_idx > max_steps:
                    done = True
                if done:
                    break


    print("Sim2Real DP on Real Push T - IoU: {}".format(max(rewards)))

    ############################ Save Result  ############################ 
    scores.append(rewards)    

    print("Eval done!")
    return scores

if __name__ == "__main__":
    main()
