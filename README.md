# Control-oriented Clustering of Visual Latent Representation (ICLR2025 Spotlight)

## Environement setup
Please run "conda env create -f environment.yml" to set up the environment required to run this code. We use Python version of 3.12.5 and Cuda 12.4.

## Prevalent Neural Collapse in Latent Representation Space

1. Train a vision-based control policy.
Here we provide a code using Resnet 18 as the vision encoder and diffusion model as the action decoder. We also provide the corresponding dataset used for training. Please run "python train_model.py" in the training folder. The default setting is training for 300 epochs(which is also our setting in the paper). It will automatically save the checkpoints which would be used to measure NC.

2. Test the trained model. 
Please run "python test_domain_18_model.py" in the evaluation_test_score folder. The default setting would evaluate saved_models at 20 different epochs during the total 300 training epochs.

3. Evaluate NC. We provide two versions of the labeling(classfication) methods for this dataset as mentioned in the paper. 
   a. Goal-based classification(input space)
   Please run "python domain_18_observe_NC_metric_input_space_labeling.py" in the observe_NC_metric_input_space_labeling folder. It would evaluate the NC metrics for the saved checkpoints at different epochs.

   b. Action-based classification(action space)
   Please run "python domain_18_observe_NC_metric_action_intention_labeling.py" in the observe_NC_metric_action_intention_labeling folder. It would evaluate the NC metrics for the saved checkpoints at different epochs.


## Control-Oriented Pretraining
In the folder of NC_pretraining, firstly run NC_pretrain.py for pretraining the vision encoder and then run NC_together.py to end-to-end train the vision encoder and diffusion model.

