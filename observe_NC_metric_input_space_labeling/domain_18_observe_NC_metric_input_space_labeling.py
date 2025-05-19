import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset

from tqdm.auto import tqdm
import argparse

import os
import yaml


# other files
from utils_input_space_labeling import *
from pusht_env import *
from models import *

import matplotlib.pyplot as plt


def main():
        
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
    seed = 2
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    num_train_demos = 500
    K = 8
    dataset_path_dir = "../dataset_domain/domain18"

    epoch_lst = [1]
    
    for i in range(1, 300):
        if i <= 80 and i % 10 == 0:
            epoch_lst.append(i)
        elif i > 80 and i % 20 ==0:
            epoch_lst.append(i)

    model_lst = [f"../training/train_model_folder/checkpoint_epoch_"+str(i)+"/vision_encoder.pth" for i in epoch_lst]


    print("Training parameters:")
    print(f"num_train_demos: {num_train_demos}")
    print(f"pred_horizon: {pred_horizon}")
    print(f"obs_horizon: {obs_horizon}")
    print(f"action_horizon: {action_horizon}")


    print("\nBaseline Mode: Train Single Diffusion Policy")

    resize_scale = 96

    print("Use default AdamW as optimizer.")
    nc1_all = []
    nc2_norm_all = []
    nc2_angle_all = []
    for model_path in model_lst:
        print(model_path)
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
        batch_size = 64
        # create dataloader
        dataloader = torch.utils.data.DataLoader(
            combined_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
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
        # cut resnet18

        nets['vision_encoder'] = vision_encoder
        

    
    

        for param in nets["vision_encoder"].parameters():
            param.requires_grad = False

        
        nets = nets.to(device)

        model_state_dict = torch.load(model_path, weights_only=True, map_location='cuda')
        nets["vision_encoder"].load_state_dict(model_state_dict)

        feature_result = []
        action_intention_result = []
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:

                # device transfer
                # data normalized in dataset
                nimage = nbatch['image'][:,:obs_horizon].to(device)
                nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]

                # encoder vision features
                image_features = nets["vision_encoder"](nimage.flatten(end_dim=1))

                action_intention = torch.squeeze(nbatch['action_intention'])
                action_intention_result.extend(action_intention.numpy().tolist())

                image_features = image_features.reshape(*nimage.shape[:2],-1)
                image_features = torch.flatten(image_features, start_dim=1)
                feature_result.append(image_features)


        feature_result = torch.concat(feature_result)
        feature_result = torch.moveaxis(feature_result, 1, 0)
        feature_result = feature_result.detach().cpu().numpy()

        action_intention_result = np.array(action_intention_result)


        feature = feature_result.T
        global_mean = np.mean(feature, axis = 0)


        label_lists = action_intention_result
        result_list = []
        for i in range(K):
            result_list.append([])
        for i, label in enumerate(label_lists):
            if label != 8:
                result_list[label].append(feature[i])
        
        cluster_mean = []
        number_samples_per_class = []
        for i in range(K):
            cluster_mean.append(np.mean(np.array(result_list[i]), axis=0) - global_mean)
            number_samples_per_class.append(len(result_list[i]))
        print('number_samples_per_class')
        print(number_samples_per_class)

        globally_centered_class_mean_result = np.array(cluster_mean)


        globally_centered_features = []
        for i in range(K):
            globally_centered_features.append(np.array(result_list[i]) - global_mean)
    
        #for nc1
        sigma_c_square = []
        number_samples_per_class = []
        for i in range(K):
            here = sum([np.square(np.linalg.norm(globally_centered_features[i][j] - globally_centered_class_mean_result[i], 2)) for j in range(globally_centered_features[i].shape[0])])
            here = here/(1.0*(globally_centered_features[i].shape[0]-1))
           
            sigma_c_square.append(here)
        
        

        within_class_cov = []
        for i in range(K):
            for j in range(K):
                if i < j:
                    within_class_cov.append((sigma_c_square[i]+sigma_c_square[j])/(2.0 * np.square(np.linalg.norm(globally_centered_class_mean_result[i] - globally_centered_class_mean_result[j], 2))))

        print("nc1")
        nc1 = np.mean(within_class_cov)
        print(nc1)



        #for nc2
        length_lst = []
        angle_lst = []
        for i in range(K):
            length = np.linalg.norm(globally_centered_class_mean_result[i])
            length_lst.append(length)


        for i in range(K):
            for j in range(K):
                if i < j:
                    angle = np.inner(globally_centered_class_mean_result[i]/np.linalg.norm(globally_centered_class_mean_result[i]), globally_centered_class_mean_result[j]/np.linalg.norm(globally_centered_class_mean_result[j]))
                    angle_lst.append(angle)

        length_lst = np.array(length_lst)
        angle_lst = np.array(angle_lst)
        nc2_equinorm = np.std(length_lst)/np.mean(length_lst)
        nc2_equiangularity = np.std(angle_lst)

        print("length_lst")
        print(length_lst)

        print("angle_lst")
        print(angle_lst)

        print("nc2 - norm")
        print(nc2_equinorm)

        print("nc2 - angularity")
        print(nc2_equiangularity)

        nc1_all.append(nc1)
        nc2_norm_all.append(nc2_equinorm)
        nc2_angle_all.append(nc2_equiangularity)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_lst, nc1_all, marker='o', linestyle='-', color='b', label='NC1')

    # Add titles and labels
    plt.xlabel('Epochs')
    plt.ylabel('CVND')

    plt.savefig(f"NC1_input_space_label.png")


    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_lst, nc2_norm_all, marker='o', linestyle='-', color='b', label='NC2_norm')

    # Add titles and labels
    plt.xlabel('Epochs')
    plt.ylabel('STD Norm')

    plt.savefig(f"NC2_norm_input_space_label.png")


    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_lst, nc2_angle_all, marker='o', linestyle='-', color='b', label='NC2_angle')

    # Add titles and labels
    plt.xlabel('Epochs')
    plt.ylabel('STD Angle')

    plt.savefig(f"NC2_angle_input_space_label.png")



if __name__ == "__main__":
    main()