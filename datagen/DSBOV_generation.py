import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from stardist import star_dist, edt_prob
import h5py
import os

##############
# Parameters #
##############

trainset_filepath = "data/DSB18/train/images/"
testset_filepath = "data/DSB18/test/images/"

trainset_GT_filepath = "data/DSB18/train/masks/"
testset_GT_filepath = "data/DSB18/test/masks/"

# all objects will be confined to this image size
img_size = (256, 256)

export_filepath = 'data/DSBOV.hdf5'

compression = 'gzip'
######################################################


def random_shift_cells(original_image, original_labels, img_size, overlap_amount=0.15, smoothing=0.8, dilation_iter=1):
    """Create an image with as many randomly shifted copies of cells from the original image to achieve enough overlap.
    
    Parameters:
    original_image -- array of shape (height, width)
    original_labels -- array of shape (num_cells, height, width)
    img_size -- tuple of height and width of new image
    overlap_amount -- percentage of cell pixels that belong to more than one cell, default 0.15
    smoothing -- amount of gaussian smoothing of the image to make it look more realistic
    dilation_iter -- number of iterations in the dilation of the labels that are used to crop the single cell images
    
    Returns: new image, array of labels
    """
    num_cells = original_labels.shape[0]
    image = np.zeros(img_size)
    labels = []

    while True:
        # iterate over all cells in original image
        for i in range(num_cells):
            # skip cell if it touches the boundary
            if (
                (np.argwhere(original_labels[i])[:, 1].min() < 2) |
                (np.argwhere(original_labels[i])[:, 0].min() < 2) |
                (np.argwhere(original_labels[i])[:, 1].max() > (original_image.shape[1] - 2)) |
                (np.argwhere(original_labels[i])[:, 0].max() > (original_image.shape[0] - 2))
                ):
                continue

            single_cell_mask = original_labels[i]
            # create a dilated mask to include more of the cell boundary area
            dilated_mask = ndimage.morphology.binary_dilation(single_cell_mask, iterations=dilation_iter)
            normalized_image = original_image #/ original_image.max()
            single_cell_image = normalized_image * (dilated_mask > 0.5)

            # shift the cell in both image and mask to the center of the new image
            x_center = (np.argwhere(single_cell_mask)[:, 1].min() + np.argwhere(single_cell_mask)[:, 1].max()) / 2
            y_center = (np.argwhere(single_cell_mask)[:, 0].min() + np.argwhere(single_cell_mask)[:, 0].max()) / 2

            single_cell_image = ndimage.shift(single_cell_image, (img_size[0] / 2 - y_center, img_size[1] / 2 - x_center))
            single_cell_mask = ndimage.shift(single_cell_mask, (img_size[0] / 2 - y_center, img_size[1] / 2 - x_center))

            # crop the to the specified size
            single_cell_image = single_cell_image[:img_size[0], :img_size[1]]
            single_cell_mask = single_cell_mask[:img_size[0], :img_size[1]]

            # random rotation
            angle = int(np.random.uniform(0, 360))

            single_cell_image = ndimage.rotate(single_cell_image, angle, reshape=False)
            single_cell_mask = ndimage.rotate(single_cell_mask, angle, reshape=False)

            # random flip
            if np.random.uniform() > 0.5:
                single_cell_image = np.flip(single_cell_image, axis=0)
                single_cell_mask = np.flip(single_cell_mask, axis=0)

            if np.random.uniform() > 0.5:
                single_cell_image = np.flip(single_cell_image, axis=0)
                single_cell_mask = np.flip(single_cell_mask, axis=0)

            # random shift
            x_shift = np.random.uniform(- img_size[1] / 2, img_size[1] / 2)
            y_shift = np.random.uniform(- img_size[0] / 2, img_size[0] / 2)

            single_cell_image = ndimage.shift(single_cell_image, (y_shift, x_shift))
            single_cell_mask = ndimage.shift(single_cell_mask, (y_shift, x_shift))

            # fix the mask
            single_cell_mask = ndimage.morphology.binary_dilation(single_cell_mask)

            image += single_cell_image
            labels.append(single_cell_mask)

        if ((np.array(labels).sum(axis=0) == 2).sum() / (np.array(labels).sum(axis=0) > 0.5).sum()) > overlap_amount:
            break

    image = ndimage.gaussian_filter(image, smoothing)
    image += np.random.normal(normalized_image[original_labels.sum(axis=0) < 0.5].mean(), normalized_image[original_labels.sum(axis=0) < 0.5].std() / 3, image.shape) * (np.array(labels).sum(axis=0) < 0.5)
    image = image / image.max()

    return image, np.array(labels)


# trainset
train_images_list = os.listdir(trainset_filepath)
train_labels_list = os.listdir(trainset_GT_filepath)

trainset_images = np.empty((len(train_images_list), img_size[0], img_size[1]))
trainset_images  = np.expand_dims(trainset_images, 1)
trainset_labels = []

for i in range(len(train_images_list)):
    original_image = plt.imread(trainset_filepath + train_images_list[i])
    integer_labels_mask = plt.imread(trainset_GT_filepath + train_labels_list[i])

    # convert mask with integer labels to mask with one channel per instance
    unique = list(np.unique(integer_labels_mask))
    unique.remove(0)
    original_labels = np.empty((len(unique), original_image.shape[0], original_image.shape[1]), dtype='int32')
    for j, k in enumerate(unique):
        original_labels[j] = (integer_labels_mask == k)
        
    shifted_image, shifted_labels = random_shift_cells(original_image, original_labels, (256, 256), 0.15)

    trainset_images[i, 0] = shifted_image
    trainset_labels.append(shifted_labels)


# test set
test_images_list = os.listdir(testset_filepath)
test_labels_list = os.listdir(testset_GT_filepath)

testset_images = np.empty((len(test_images_list), img_size[0], img_size[1]))
testset_images = np.expand_dims(testset_images, 1)
testset_labels = []

for i in range(len(test_images_list)):
    original_image = plt.imread(testset_filepath + test_images_list[i])
    integer_labels_mask = plt.imread(testset_GT_filepath + test_labels_list[i])

    # convert mask with integer labels to mask with one channel per instance
    unique = list(np.unique(integer_labels_mask))
    unique.remove(0)
    original_labels = np.empty((len(unique), original_image.shape[0], original_image.shape[1]), dtype='int32')
    for j, k in enumerate(unique):
        original_labels[j] = (integer_labels_mask == k)
        
    shifted_image, shifted_labels = random_shift_cells(original_image, original_labels, (256, 256), 0.15)

    testset_images[i, 0] = shifted_image
    testset_labels.append(shifted_labels)

# saving created images
f = h5py.File(export_filepath, 'w')
trainset_labels_group = f.create_group("trainset_labels")
testset_labels_group = f.create_group("testset_labels")

f.create_dataset("trainset", data=trainset_images, compression=compression)
for i in range(len(trainset_labels)):
    trainset_labels_group.create_dataset(str(i), data=trainset_labels[i], compression=compression)

f.create_dataset("testset", data=testset_images, compression=compression)
for i in range(len(testset_labels)):
    testset_labels_group.create_dataset(str(i), data=testset_labels[i], compression=compression)