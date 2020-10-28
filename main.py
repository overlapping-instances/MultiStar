from shutil import copytree
import os
import re
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import trainloop
import sampling
import loss
import dataset
import multitaskmodel
import utils

####################################
############ Parameters ############
####################################

# dataset
dataset_path = "data/DSBOV_overlap_images.hdf5"
trainset_dataset_name = "trainset"
trainset_labels_group_name = "trainset_labels"
testset_dataset_name = "testset"
testset_labels_group_name = "testset_labels"

# number of star distance directions
num_rays = 32

# elastic deformation (augmentation)
elastic_deform_sigma = 7
elastic_deform_points = 3
zoom_factor = 1.2

# data loader
train_batch_size = 4
test_batch_size = 1
num_workers = 15 # heavy preprocessing requires many workers

# model parameters
out_channels = 256
fmaps = (16, 32, 64, 128, 256)

# optimizer learning rate
lr = 1e-4

# training
epochs = 40
plot_every = 10
evaluate_every = 1
save_every = 5

# NMS sampling in tensorboard
num_proposals = 500
iou_thres = 0.1
min_objprob = 0.3

#############################################################################
#############################################################################

# gpu selection
gpu_id = str(input("Select gpu: "))
device = torch.device("cuda:" + gpu_id)

# automatic experiment name and path determination
exp_name = "run1"
while os.path.exists(os.path.join('experiments/results', exp_name)):
    exp_name = "run" + str(int(re.findall('\d+', exp_name)[0]) + 1)


trainset = dataset.Dataset(
    path=dataset_path,
    images_dataset_name=trainset_dataset_name,
    labels_group_name=trainset_labels_group_name
    )

testset = dataset.Dataset(
    path=dataset_path,
    images_dataset_name=testset_dataset_name,
    labels_group_name=testset_labels_group_name,
    )

# these images are used to plot predictions in tensorboard
plot_trainset = trainset.get_plot_images(2)
plot_testset = testset.get_plot_images(2)

trainloader = DataLoader(
    dataset=trainset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers
    )

testloader = DataLoader(
    dataset=testset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=num_workers
    )

model = multitaskmodel.MultitaskModel(
    out_channels=out_channels,
    fmaps=fmaps,
    )

# wrapper for homoscedastic uncertainty loss
mtl = loss.MultiTaskLossWrapper(model=model)

optimizer = torch.optim.Adam(mtl.parameters(), lr=lr)

# make experiment results directory
exp_results_path = os.path.join('experiments/results', exp_name)
os.makedirs(exp_results_path, exist_ok=False)

# save all relevant scripts in experiment results directory
copytree('experiments/dev', os.path.join(exp_results_path, 'scripts'))

trainer = trainloop.Trainer(
    exp_path=exp_results_path,
    model=model,
    mtl=mtl,
    optimizer=optimizer,
    trainloader=trainloader,
    testloader=testloader,
    device=device,
    num_proposals=num_proposals,
    iou_thres=iou_thres,
    min_objprob=min_objprob,
    plot_trainset=plot_trainset,
    plot_testset=plot_testset,
    plot_every=plot_every,
    evaluate_every=evaluate_every,
    save_every=save_every,
    )

trainer.train_model(epochs=epochs)

# save final model
torch.save(trainer.model.state_dict(), os.path.join(os.path.join(exp_results_path, 'state_dict.pt')))