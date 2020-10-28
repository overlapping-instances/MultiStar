import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from stardist import star_dist, edt_prob
import h5py
import os
from scipy.io import loadmat


##############
# Parameters #
##############

trainset_filepath = 'data/ISBI15/Training/'
testset_filepath = 'data/ISBI15/Test/'

trainset_GT_filepath = 'data/ISBI15/Training_annotations'
testset_GT_filepath = 'data/ISBI15/Test_annotations.mat'

export_filepath = 'data/ISBI15_images.hdf5'

compression = 'gzip'
######################################################


def create_original_images(path):
    """Load the original images from path and put them into one array.

    Parameters:
    path -- path to look for the images

    Returns: array of shape (num_images, height, width)
    """
    
    images_list = sorted(os.listdir(path))
    
    num_images = len(images_list)
    height = plt.imread(os.path.join(path, images_list[0])).shape[0]
    width = plt.imread(os.path.join(path, images_list[0])).shape[1]
    
    images = np.empty((num_images, height, width), dtype='float32')
    
    for i, x in enumerate(images_list):
        images[i] = plt.imread(os.path.join(path, x))
        images[i] = images[i] / images[i].max()

    images = np.expand_dims(images, 1)
    
    return images

def create_masks_png(path):
    """Load the masks from path and put them into a list of binary images.

    Parameters:
    path -- path to look for the masks

    Returns: list with entries: boolean array of shape (num_cells, height, width)
    """
    
    masks = []
    
    for x in sorted([x[0] for x in os.walk(path)][1:]):

        masks_list = os.listdir(x)

        num_masks = len(masks_list)
        height = plt.imread(os.path.join(x, masks_list[0])).shape[0]
        width = plt.imread(os.path.join(x, masks_list[0])).shape[1]

        masks.append(np.empty((num_masks, height, width), dtype='float32'))

        for i, y in enumerate(masks_list):
            masks[-1][i] = plt.imread(os.path.join(x, y))

    return masks

def create_masks_mat(path):
    """Load the masks from the .mat file at path and put them in a list of binary images.

    Parameters:
    path -- path of the .mat file with the annotations

    Returns: list with entries: boolean array of shape (num_cells, height, width)
    """

    masks = []
    
    data = loadmat(path)["AnnotationTest"]
    
    num_masks = data.shape[0]
    height = data[0][0][0][0].shape[0]
    width = data[0][0][0][0].shape[1]
    
    for i in range(num_masks):
        num_cells = data[i][0].shape[0]
        
        masks.append(np.empty((num_cells, height, width), dtype='float32'))
        
        for j in range(num_cells):
            masks[-1][j] = data[i][0][j][0]
            
    return masks

train_images = create_original_images(trainset_filepath)
test_images = create_original_images(testset_filepath)

train_labels = create_masks_png(trainset_GT_filepath)
test_labels = create_masks_mat(testset_GT_filepath)

f = h5py.File(export_filepath, 'w')

f.create_dataset('trainset', data=train_images, compression=compression)

trainset_labels_group = f.create_group("trainset_labels")
for i in range(len(train_labels)):
    trainset_labels_group.create_dataset(name=str(i), data=train_labels[i], compression=compression)

f.create_dataset('testset', data=test_images, compression=compression)

testset_labels_group = f.create_group("testset_labels")
for i in range(len(test_labels)):
    testset_labels_group.create_dataset(name=str(i), data=test_labels[i], compression=compression)