import torch
import h5py
import numpy as np
from scipy import ndimage
import elasticdeform
from stardist import edt_prob, star_dist
from skimage.transform import resize
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        images_dataset_name,
        labels_group_name,
        min_max_value=None,
        num_rays=32,
        use_only=None,
        use_transforms=False,
        elastic_deform_sigma=10,
        elastic_deform_points=3,
        zoom_factor=1.1,
        crop_size=None
        ):
        """Dataset for single data source.
        
        Parameters:
        path -- location of hdf5 file
        images_dataset_name -- hdf5 dataset name of images
        labels_group_name -- hdf5 group name of image labels
        min_max_value -- tuple with minimum and maximum values for scaling, derived from data if None
        num_rays -- number of polygon distances, default 32
        use_only -- number of images to use from the dataset, useful for faster debugging, default all images
        use_transforms -- boolean, decides whether transforms and crops are applied or not
        elastic_deform_sigma -- higher value leads to stronger deformation
        elastic_deform_points -- number of grid points in the elastic deformation
        zoom_factor -- zoom factor for cropping after elastic deformation
        crop_size -- tuple with size for random crops, no crops is None
        """

        self.elastic_deform_sigma = elastic_deform_sigma
        self.elastic_deform_points = elastic_deform_points
        self.zoom_factor = zoom_factor
        self.num_rays = num_rays
        self.crop_size = crop_size
        self.use_transforms = use_transforms

        # load data
        with h5py.File(path, 'r') as f:
            self.images = f[images_dataset_name][:use_only].astype("float32")
            self.labels = []
            for i in range(len(self.images)):
                self.labels.append(f[labels_group_name][str(i)][()].astype("float32"))
        
        # determine normalization
        self.min_max_value = min_max_value
        if self.min_max_value == None:
            self.min_max_value = (self.images.min(), self.images.max())


    def random_flips(self, image, labels):
        flip_axes = [1, 2, (1, 2)]
        flip_axis = flip_axes[np.random.choice([0, 1, 2])]
        if np.random.random(1).item() > 0.25:
            return np.flip(image, axis=flip_axis), np.flip(labels, axis=flip_axis)

        return image, labels

    def random_rotation(self, image, labels):
        num_rotations = np.random.choice([0, 1, 2, 3])

        return np.rot90(image, num_rotations, axes=(1, 2)), np.rot90(labels, num_rotations, axes=(1, 2))

    def random_elastic_deform(self, image, labels, sigma, points):
        cropy, cropx = image.shape[1:]

        image = ndimage.zoom(image, (1, self.zoom_factor, self.zoom_factor), order=1)
        labels = ndimage.zoom(labels, (1, self.zoom_factor, self.zoom_factor), order=1)

        y,x = image.shape[1:]
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)  
        
        [image, labels] = elasticdeform.deform_random_grid(X=[image, labels], axis=(1, 2), sigma=sigma, points=points, order=1, crop=[slice(starty, (starty+cropy)), slice(startx, (startx+cropx))], mode="mirror")
        
        return image, labels

    def transform(self, image, labels):
        """Apply random flips, rotations and elastic deformations.

        Parameters:
        image -- array of shape (1, height, width) with image
        labels -- array of shape (num_cells, height, width) with cell masks

        Returns: transformed image, transformed labels
        """

        if not self.crop_size is None:
            image, labels = self.random_crop(image, labels, self.crop_size)

        image, labels = self.random_flips(image, labels)
        image, labels = self.random_rotation(image, labels)
        image, labels = self.random_elastic_deform(image, labels, self.elastic_deform_sigma, self.elastic_deform_points)

        return image, labels

    def get_overlap(self, labels):
        overlap = (labels.sum(axis=0) > 1.5).astype("float32")
        
        return np.expand_dims(overlap, 0)

    def get_stardistances(self, labels):
        stardistances = np.zeros((self.num_rays, labels.shape[1], labels.shape[2]), dtype="float32")
        
        for i in range(labels.shape[0]):
            stardistances += star_dist(labels[i], self.num_rays).transpose(2, 0, 1)

        stardistances[:, labels.sum(axis=0) > 1.5] = 0

        return stardistances

    def get_objectprobs(self, labels):
        objectprobs = np.zeros((1, labels.shape[1], labels.shape[2]), dtype="float32")

        for i in range(labels.shape[0]):
            objectprobs += edt_prob(labels[i].astype("int32"))

        objectprobs[:, labels.sum(axis=0) > 1.5] = 0

        return objectprobs

    def normalize_image(self, image):
        return (image - self.min_max_value[0]) / (self.min_max_value[1] - self.min_max_value[0])

    def get_plot_images(self, num_images):
        """Get a batch of images, labels, overlap, stardistances, object probabilities for plotting.

        Parameters:
        num_images -- batch size

        Returns: images array (num_images, 1, height, width), labels list with arrays of shape (num_cells, height, width),
        overlap array (num_images, 1, height, width), star distances array (num_images, 32, height, width), object probabilities
        array (num_images, 1, height, width)
        """

        # use first images in dataset 
        plot_images = self.images[:num_images]
        plot_labels = self.labels[:num_images]

        overlap = np.zeros((num_images, 1, plot_images.shape[2], plot_images.shape[3]), dtype="float32")
        stardistances = np.zeros((num_images, self.num_rays, plot_images.shape[2], plot_images.shape[3]), dtype="float32")
        objectprobs = np.zeros((num_images, 1, plot_images.shape[2], plot_images.shape[3]), dtype="float32")
        
        # iterate over images and compute features and normalized images
        for i in range(num_images):
            overlap[i] = self.get_overlap(plot_labels[i])
            stardistances[i] = self.get_stardistances(plot_labels[i])
            objectprobs[i] = self.get_objectprobs(plot_labels[i])
            plot_images[i] = self.normalize_image(plot_images[i])

        plot_images = torch.from_numpy(plot_images)
        overlap = torch.from_numpy(overlap)
        stardistances = torch.from_numpy(stardistances)
        objectprobs = torch.from_numpy(objectprobs)

        return plot_images, plot_labels, overlap, stardistances, objectprobs

    def random_crop(self, image, labels, size):
        y_start = np.random.randint(0, image.shape[1] - size[0] - 1)
        x_start = np.random.randint(0, image.shape[2] - size[1] - 1)

        return image[:, y_start:y_start+size[0], x_start:x_start+size[1]], labels[:, y_start:y_start+size[0], x_start:x_start+size[1]]
        
    def __getitem__(self, index):
        if self.use_transforms:
            image, labels = self.transform(self.images[index], self.labels[index])
        else:
            image = self.images[index]
            labels = self.labels[index]
        image = self.normalize_image(image)
        overlap = self.get_overlap(labels)
        stardistances = self.get_stardistances(labels)
        objectprobs = self.get_objectprobs(labels)

        return torch.from_numpy(image), torch.from_numpy(overlap), torch.from_numpy(stardistances), torch.from_numpy(objectprobs)

    def __len__(self):
        return self.images.shape[0]


class DatasetPlus(torch.utils.data.Dataset):
    def __init__(
        self,
        path1_2_3,
        path4,
        images1_dataset_name,
        images2_dataset_name,
        images3_dataset_name,
        images4_dataset_name,
        labels1_group_name,
        labels2_group_name,
        labels3_group_name,
        labels4_group_name,
        min_max_value=None,
        num_rays=32,
        use_transforms=False,
        elastic_deform_sigma=10,
        elastic_deform_points=3,
        zoom_factor=1.1,
        crop_size=None
        ):
        """Special class for dataset made of ISBI14 Train, ISBI14 Test90 and ISBI15 Train
        
        Parameters:
        path -- location of hdf5 file
        images_dataset_name -- hdf5 dataset name of images
        labels_group_name -- hdf5 group name of image labels
        min_max_value -- tuple with minimum and maximum values for scaling, derived from data if None
        num_rays -- number of polygon distances, default 32
        use_only -- number of images to use from the dataset, useful for faster debugging, default all images
        use_transforms -- boolean, decides whether transforms and crops are applied or not
        elastic_deform_sigma -- higher value leads to stronger deformation
        elastic_deform_points -- number of grid points in the elastic deformation
        zoom_factor -- zoom factor for cropping after elastic deformation
        crop_size -- tuple with size for random crops, no crops is None
        """

        self.elastic_deform_sigma = elastic_deform_sigma
        self.elastic_deform_points = elastic_deform_points
        self.zoom_factor = zoom_factor
        self.num_rays = num_rays
        self.crop_size = crop_size
        self.use_transforms = use_transforms

        # load data
        with h5py.File(path1_2_3, 'r') as f:
            # isbi14 data, already has correct shape
            images1 = f[images1_dataset_name][()]
            labels1 = []
            for i in range(len(images1)):
                labels1.append(f[labels1_group_name][str(i)][()].astype("float32"))
            images2 = f[images2_dataset_name][()]
            labels2 = []
            for i in range(len(images2)):
                labels2.append(f[labels2_group_name][str(i)][()].astype("float32"))

            images3 = f[images3_dataset_name][()]
            labels3 = []
            for i in range(len(images3)):
                labels3.append(f[labels3_group_name][str(i)][()].astype("float32"))

        with h5py.File(path4, 'r') as f:
            # isbi15 data, 4 times as large images as isbi14
            images4 = []
            labels4 = []
            images_large = f[images4_dataset_name][()]
            labels_large = []

            for i in range(len(images_large)):
                labels_large.append(f[labels4_group_name][str(i)][:])

            for i in range(images_large.shape[0]):
                tiles, tiles_labels = self.crop_four(images_large[i, 0], labels_large[i])
                images4.append(tiles)
                labels4 += tiles_labels

            images4 = np.vstack(images4)

        # put everything together
        self.images = np.concatenate((images1, images2, images3, images4), axis=0).astype("float32")
        self.labels = labels1 + labels2 + labels3 + labels4

        self.min_max_value = min_max_value
        if self.min_max_value == None:
            self.min_max_value = (self.images.min(), self.images.max())

    def crop_four(self, image, labels):
        tiles = np.empty((4, 1, int(image.shape[0] / 2), int(image.shape[1] / 2)))
        tiles[0, 0] = image[:int(image.shape[0] / 2), :int(image.shape[1] / 2)]
        tiles[1, 0] = image[:int(image.shape[0] / 2), int(image.shape[1] / 2):]
        tiles[2, 0] = image[int(image.shape[0] / 2):, :int(image.shape[1] / 2)]
        tiles[3, 0] = image[int(image.shape[0] / 2):, int(image.shape[1] / 2):]

        tiles_labels = []
        tiles_labels.append(labels[:, :int(labels.shape[1] / 2), :int(labels.shape[2] / 2)].astype("float32"))
        tiles_labels.append(labels[:, :int(labels.shape[1] / 2), int(labels.shape[2] / 2):].astype("float32"))
        tiles_labels.append(labels[:, int(labels.shape[1] / 2):, :int(labels.shape[2] / 2)].astype("float32"))
        tiles_labels.append(labels[:, int(labels.shape[1] / 2):, int(labels.shape[2] / 2):].astype("float32"))

        for i in range(len(tiles_labels)):
            tiles_labels[i] = np.delete(tiles_labels[i], np.where(1 - tiles_labels[i].any(axis=(1, 2))), axis=0)
            
        return tiles, tiles_labels

    def random_flips(self, image, labels):
        flip_axes = [1, 2, (1, 2)]
        flip_axis = flip_axes[np.random.choice([0, 1, 2])]
        if np.random.random(1).item() > 0.25:
            return np.flip(image, axis=flip_axis), np.flip(labels, axis=flip_axis)

        return image, labels

    def random_rotation(self, image, labels):
        num_rotations = np.random.choice([0, 1, 2, 3])

        return np.rot90(image, num_rotations, axes=(1, 2)), np.rot90(labels, num_rotations, axes=(1, 2))

    def random_elastic_deform(self, image, labels, sigma, points):
        cropy, cropx = image.shape[1:]

        image = ndimage.zoom(image, (1, self.zoom_factor, self.zoom_factor), order=1)
        labels = ndimage.zoom(labels, (1, self.zoom_factor, self.zoom_factor), order=1)

        y,x = image.shape[1:]
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)  
        
        [image, labels] = elasticdeform.deform_random_grid(X=[image, labels], axis=(1, 2), sigma=sigma, points=points, order=1, crop=[slice(starty, (starty+cropy)), slice(startx, (startx+cropx))])
        
        return image, labels

    def transform(self, image, labels):
        """Apply random flips, rotations and elastic deformations.

        Parameters:
        image -- array of shape (1, height, width) with image
        labels -- array of shape (num_cells, height, width) with cell masks

        Returns: transformed image, transformed labels
        """

        if not self.crop_size is None:
            image, labels = self.random_crop(image, labels, self.crop_size)

        image, labels = self.random_flips(image, labels)
        image, labels = self.random_rotation(image, labels)
        image, labels = self.random_elastic_deform(image, labels, self.elastic_deform_sigma, self.elastic_deform_points)

        return image, labels

    def get_overlap(self, labels):
        overlap = (labels.sum(axis=0) > 1.5).astype("float32")
        
        return np.expand_dims(overlap, 0)

    def get_stardistances(self, labels):
        stardistances = np.zeros((self.num_rays, labels.shape[1], labels.shape[2]), dtype="float32")
        
        for i in range(labels.shape[0]):
            stardistances += star_dist(labels[i], self.num_rays).transpose(2, 0, 1)

        stardistances[:, labels.sum(axis=0) > 1.5] = -1

        return stardistances

    def get_objectprobs(self, labels):
        objectprobs = np.zeros((1, labels.shape[1], labels.shape[2]), dtype="float32")

        for i in range(labels.shape[0]):
            objectprobs += edt_prob(labels[i].astype("int32"))

        objectprobs[:, labels.sum(axis=0) > 1.5] = -1

        return objectprobs

    def normalize_image(self, image):
        return (image - self.min_max_value[0]) / (self.min_max_value[1] - self.min_max_value[0])

    def get_plot_images(self, num_images):
        """Get a batch of images, labels, overlap, stardistances, object probabilities for plotting.

        Parameters:
        num_images -- batch size

        Returns: images array (num_images, 1, height, width), labels list with arrays of shape (num_cells, height, width),
        overlap array (num_images, 1, height, width), star distances array (num_images, 32, height, width), object probabilities
        array (num_images, 1, height, width)
        """

        # use first images in dataset 
        plot_images = self.images[:num_images]
        plot_labels = self.labels[:num_images]

        overlap = np.zeros((num_images, 1, plot_images.shape[2], plot_images.shape[3]), dtype="float32")
        stardistances = np.zeros((num_images, self.num_rays, plot_images.shape[2], plot_images.shape[3]), dtype="float32")
        objectprobs = np.zeros((num_images, 1, plot_images.shape[2], plot_images.shape[3]), dtype="float32")
        
        # iterate over images and compute features and normalized images
        for i in range(num_images):
            overlap[i] = self.get_overlap(plot_labels[i])
            stardistances[i] = self.get_stardistances(plot_labels[i])
            objectprobs[i] = self.get_objectprobs(plot_labels[i])
            plot_images[i] = self.normalize_image(plot_images[i])

        plot_images = torch.from_numpy(plot_images)
        overlap = torch.from_numpy(overlap)
        stardistances = torch.from_numpy(stardistances)
        objectprobs = torch.from_numpy(objectprobs)

        return plot_images, plot_labels, overlap, stardistances, objectprobs

    def random_crop(self, image, labels, size):
        y_start = np.random.randint(0, image.shape[1] - size[0] - 1)
        x_start = np.random.randint(0, image.shape[2] - size[1] - 1)

        return image[:, y_start:y_start+size[0], x_start:x_start+size[1]], labels[:, y_start:y_start+size[0], x_start:x_start+size[1]]
        
    def __getitem__(self, index):
        if self.use_transforms:
            image, labels = self.transform(self.images[index], self.labels[index])
        else:
            image = self.images[index]
            labels = self.labels[index]
        image = self.normalize_image(image)
        overlap = self.get_overlap(labels)
        stardistances = self.get_stardistances(labels)
        objectprobs = self.get_objectprobs(labels)

        return torch.from_numpy(image), torch.from_numpy(overlap), torch.from_numpy(stardistances), torch.from_numpy(objectprobs)

    def __len__(self):
        return self.images.shape[0]
