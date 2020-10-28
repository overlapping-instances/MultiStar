import sampling
import loss
import utils
import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast

class Trainer(object):
    def __init__(
        self,
        exp_path,
        model,
        mtl,
        optimizer,
        trainloader,
        testloader,
        device,
        num_proposals,
        iou_thres,
        min_objprob,
        plot_trainset,
        plot_testset,
        overlap_weight=1,
        stardist_weight=1,
        objprob_weight=1,
        plot_every=1,
        evaluate_every=1,
        save_every=20,
        ):
        """Trainer class for model training.
        
        Parameters:
        exp_path -- experiment path, tensorboard uses for logging
        model -- multitaskmodel.MultitaskModel instance
        mtl -- instance of multitask loss wrapper
        optimizer -- optimizer instance
        trainloader -- DataLoader instance with training data
        testloader -- DataLoader instance with test data
        device -- cuda device
        num_proposals -- number of proposals to use in non-maximum suppression in segmentation shown in tensorboard
        iou_thres -- intersection over union threshold to use in non-maximum suppression in segmentation shown in tensorboard
        min_objprob -- minimum required object probability to sample pixel position in segmentation shown in tensorboard
        plot_trainset -- list of images, labels, overlap, stardistances, object probabilities for a training batch, will be used in tensorboard plots
        plot_testset -- list of images, labels, overlap, stardistances, object probabilities for a test batch, will be used in tensorboard plots
        plot_every -- number of epochs after which tensorboard plots are made
        evaluate_every -- number of epochs after which evaluation on testset occurs
        save_every -- number of epochs after which model is saved as state dict
        """

        self.model = model
        self.mtl = mtl
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.num_proposals = num_proposals
        self.iou_thres = iou_thres
        self.writer = SummaryWriter(exp_path)
        self.exp_path = exp_path
        self.min_objprob = min_objprob
        self.plot_trainset = plot_trainset
        self.plot_testset = plot_testset
        self.plot_every = plot_every
        self.evaluate_every = evaluate_every
        self.save_every = save_every

    def train_model(self, epochs=1):
        """Train the model for a given number of epochs.

        Parameters:
        epochs -- number of training epochs, default 1
        """

        # transfer model parameters to GPU
        model = self.model.to(device=self.device, dtype=torch.float32)
        mtl = self.mtl.to(device=self.device, dtype=torch.float32)

        # number of batches
        num_iter_train = len(self.trainloader)

        scaler = GradScaler()

        for e in range(epochs):
            for i, (images, overlap, stardist, objprob) in enumerate(self.trainloader):
                model.train()

                self.optimizer.zero_grad()

                # transfer data to GPU
                images = images.to(device=self.device, dtype=torch.float32)
                overlap = overlap.to(device=self.device, dtype=torch.float32)
                stardist = stardist.to(device=self.device, dtype=torch.float32)
                objprob = objprob.to(device=self.device, dtype=torch.float32)

                with autocast():
                    total_loss, loss_overlap, loss_stardist, loss_objprob, prec_overlap, prec_stardist, prec_objprob = mtl(images, overlap, stardist, objprob)

                # backpropagation and parameter update
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)

                scaler.update()

                # log losses on tensorboard
                self.writer.add_scalar("Training set total loss", total_loss.item(), i + e * num_iter_train)
                self.writer.add_scalar("Training set overlap loss", loss_overlap.item(), i + e * num_iter_train)
                self.writer.add_scalar("Training set stardistances loss", loss_stardist.item(), i + e * num_iter_train)
                self.writer.add_scalar("Training set object probabilities loss", loss_objprob.item(), i + e * num_iter_train)
                self.writer.add_scalar("Training set overlap precision", prec_overlap, i + e * num_iter_train)
                self.writer.add_scalar("Training set stardistances precision", prec_stardist, i + e * num_iter_train)
                self.writer.add_scalar("Training set object probabilities precision", prec_objprob, i + e * num_iter_train)

                print("Epoch: %s Iteration: %s ---> Total loss: %s" % (str(e), str(i), str(total_loss.item())))

                del images, overlap, stardist, objprob

            # evaluate on testset
            if (e % self.evaluate_every) == 0:
                self.evaluate_on_testset(e * num_iter_train, model)

            if ((e % self.plot_every) == 0) & (e > 0):
                # plots of training set
                self.make_plots(
                    images=self.plot_trainset[0],
                    labels=self.plot_trainset[1],
                    overlap=self.plot_trainset[2],
                    stardist=self.plot_trainset[3],
                    objprob=self.plot_trainset[4],
                    model=model,
                    plot_name='Training set',
                    epoch=e,
                    num_proposals=self.num_proposals,
                    iou_thres=self.iou_thres,
                    min_objprob=self.min_objprob
                    )

                # plots of test set
                self.make_plots(
                    images=self.plot_testset[0],
                    labels=self.plot_testset[1],
                    overlap=self.plot_testset[2],
                    stardist=self.plot_testset[3],
                    objprob=self.plot_testset[4],
                    model=model,
                    plot_name='Test set',
                    epoch=e,
                    num_proposals=self.num_proposals,
                    iou_thres=self.iou_thres,
                    min_objprob=self.min_objprob
                    )

            # save model
            if ((e % self.save_every) == 0) & (e > 0):
                torch.save(model.state_dict(), os.path.join(os.path.join(self.exp_path, 'state_dict_epoch%s.pt' %e)))

    def make_plots(
        self,
        images,
        labels,
        overlap,
        stardist,
        objprob,
        model,
        plot_name,
        epoch,
        num_proposals,
        iou_thres,
        min_objprob
        ):
        """Make plots of the predicted/true overlap, star distances, object probabilities and segmentation on tensorboard.

        Parameters:
        images -- array of shape (num_images, 1, height, width) with normalized images
        labels -- list of arrays, with each array of shape (num_cells, height, width) with the cell masks
        overlap -- array of shape (num_images, 1, height, width) with overlap values
        stardist -- array of shape (num_images, 32, height, width) with the 32 star distance values at every pixel
        objprob -- array of shape (num_images, 1, height, width) with the object probabilities
        model -- model instance
        plot_name -- string, training set / test set
        epoch -- training epoch
        num_proposals -- number of proposals to use in the non-maximum suppression of the segmentation generation
        iou_thres -- intersection over union threshold to use in the non-maximum suppression of the segmentation generation
        min_objprob -- minimum required object probability to sample a pixel in the segmentation generation
        """

        # transfer data to gpu
        images = images.to(device=self.device, dtype=torch.float32)
        overlap = overlap.to(device=self.device, dtype=torch.float32)
        stardist = stardist.to(device=self.device, dtype=torch.float32)
        objprob = objprob.to(device=self.device, dtype=torch.float32)

        # make predictions with model
        pred_overlap, pred_stardist, pred_objprob = model(images)

        pred_overlap = torch.sigmoid(pred_overlap)
        pred_objprob = torch.sigmoid(pred_objprob)

        # transfer data and predictions back to cpu and convert to numpy arrays, to plot them
        images = images.cpu().detach().numpy()
        overlap = overlap.cpu().detach().numpy()
        stardist = stardist.cpu().detach().numpy()
        objprob = objprob.cpu().detach().numpy()
        pred_overlap = pred_overlap.cpu().detach().numpy()
        pred_stardist = pred_stardist.cpu().detach().numpy()
        pred_objprob = pred_objprob.cpu().detach().numpy()

        # generate different plots on tensorboard
        self.plot_overlap(images, pred_overlap, overlap, epoch, plot_name)
        self.plot_stardist(images, pred_stardist, stardist, objprob, epoch, plot_name)
        self.plot_objprob(images, pred_objprob, objprob, overlap, epoch, plot_name)
        self.plot_segmentation(labels, images, pred_overlap, pred_stardist, pred_objprob, num_proposals, iou_thres, min_objprob, epoch, plot_name)

        del images, overlap, stardist, objprob, pred_overlap, pred_objprob, pred_stardist

    def plot_overlap(self, images, pred_overlap, overlap, epoch, plot_name):
        """Plot the original images, the predicted and the true overlap.

        Parameters:
        images -- array of shape (num_images, 1, height, width) with images
        pred_overlap -- array of shape (num_images, 1, height, width) with predicted overlap
        overlap -- array of shape (num_images, 1, height, width) with true overlap
        epoch -- training epoch
        plot_name -- string, training set / test set
        """

        fig, axs = plt.subplots(images.shape[0], 3, figsize=(13,13))

        axs[0, 0].set_title("Input image")
        axs[0, 1].set_title("Overlap prediction")
        axs[0, 2].set_title("Overlap gt")

        for i in range(images.shape[0]):
            axs[i, 0].imshow(images[i, 0], cmap='gray')
            axs[i, 1].imshow(pred_overlap[i, 0], cmap='gray', vmin=0, vmax=1)
            axs[i, 2].imshow(overlap[i, 0], cmap='gray', vmin=0, vmax=1)

        plt.tight_layout()
        self.writer.add_figure(plot_name + " overlap", fig, epoch)

    def plot_stardist(self, images, pred_stardist, stardist, objprob, epoch, plot_name):
        """Plot the original images, the predicted and the true star distances.
        Mask pixels where the object probability is zero or where objects overlap.

        Parameters:
        images -- array of shape (num_images, 1, height, width) with images
        pred_stardist -- array of shape (num_images, num_rays, height, width) with predicted star distances
        stardist -- array of shape (num_images, num_rays, height, width) with true star distances
        objprobs -- array of shape (num_images, 1, height, width) with true object probabilities, to mask the image
        epoch -- training epoch
        plot_name -- string, training set / test set
        """

        pred_stardist[np.repeat(objprob, stardist.shape[1], 1) < 1e-3] = 0
        
        fig, axs = plt.subplots(images.shape[0], 3, figsize=(13,13))

        axs[0, 0].set_title("Input image")
        axs[0, 1].set_title("Stardistances prediction \n (0th ray)")
        axs[0, 2].set_title("Stardistances gt \n (0th ray)")

        for i in range(images.shape[0]):
            axs[i, 0].imshow(images[i, 0], cmap='gray')
            axs[i, 1].imshow(pred_stardist[i, 0], cmap='gray', vmin=0)
            axs[i, 2].imshow(stardist[i, 0], cmap='gray', vmin=0)

        plt.tight_layout()
        self.writer.add_figure(plot_name + " stardistances", fig, epoch)

    def plot_objprob(self, images, pred_objprob, objprob, overlap, epoch, plot_name):
        """Plot the original images, the predicted and the true object probabilities.

        Maks pixels where objects overlap.
        Parameters:
        images -- array of shape (num_images, 1, height, width) with images
        pred_objprob -- array of shape (num_images, 1, height, width) with predicted object probabilities
        objprob -- array of shape (num_images, 1, height, width) with true object probabilities
        overlap -- array of shape (num_images, 1, height, width) with true overlap, to mask the image
        epoch -- training epoch
        plot_name -- string, training set / test set
        """

        pred_objprob[overlap > 0.5] = 0
        
        fig, axs = plt.subplots(images.shape[0], 3, figsize=(13,13))

        axs[0, 0].set_title("Input image")
        axs[0, 1].set_title("Object probabilities prediction")
        axs[0, 2].set_title("Object probabilities gt")

        for i in range(images.shape[0]):
            axs[i, 0].imshow(images[i, 0], cmap='gray')
            axs[i, 1].imshow(pred_objprob[i, 0], cmap='gray', vmin=0, vmax=1)
            axs[i, 2].imshow(objprob[i, 0], cmap='gray', vmin=0, vmax=1)

        plt.tight_layout()
        self.writer.add_figure(plot_name + " object probabilities", fig, epoch)

    def plot_segmentation(self, labels, images, pred_overlap, pred_stardist, pred_objprob, num_proposals, iou_thres, min_objprob, epoch, plot_name):
        """Plot the original images, the predicted and the true segmentation.

        Additionally, the sampling position is shown for every predicted instance.
        Parameters:
        labels -- list of arrays of shapes (num_cells, height, width) with cell masks
        images -- array of shape (num_images, 1, height, width) with images
        pred_overlap -- array of shape (num_images, 1, height, width) with predicted overlap
        pred_stardist -- array of shape (num_images, 32, height, width) with predicted star distances
        pred_objprob -- array of shape (num_images, 1, height, width) with predicted object probabilities
        num_proposals -- number of proposals to generate for the segmentation
        iou_thres -- intersection over union threshold on two proposals above which the less confident is suppressed
        min_objprob -- minimum required object probability to sample a pixel position
        epoch -- training epoch
        plot_name -- string, training set / test set
        """

        fig, axs = plt.subplots(images.shape[0], 3, figsize=(13,13))

        axs[0, 0].set_title("Input image")
        axs[0, 1].set_title("Segmentation prediction \n with sampling positions")
        axs[0, 2].set_title("Segmentation gt")

        for i in range(images.shape[0]):
            polygons_pred, center_coordinates = sampling.nms(
                pred_overlap[i, 0],    
                pred_stardist[i],
                pred_objprob[i, 0],
                num_proposals=num_proposals,
                iou_thres=iou_thres,
                min_objprob=min_objprob
                )

            axs[i, 0].imshow(images[i, 0], cmap='gray')
            utils.plot_contours(polygons_pred, images[i, 0], axs[i, 1], center_coordinates)
            utils.plot_contours(labels[i], images[i, 0], axs[i, 2])

        plt.tight_layout()
        self.writer.add_figure(plot_name + " segmentation", fig, epoch)

    def evaluate_on_testset(self, step, model):
        """Evaluate the model on the testset.

        Parameters:
        step -- total number of iterations done so far (epochs * iterations per epoch)
        model -- model instance
        """

        model.eval()

        # initialize losses
        loss_total = 0
        loss_overlap = 0
        loss_stardist = 0
        loss_objprob = 0
        prec_overlap = 0
        prec_stardist = 0
        prec_objprob = 0

        # number of batches in testset
        num_iter_test = len(self.testloader)

        with torch.no_grad():
            for i, (images, overlap, stardist, objprob) in enumerate(self.testloader):

                # transfer data to gpu
                images = images.to(device=self.device, dtype=torch.float32)
                overlap = overlap.to(device=self.device, dtype=torch.float32)
                stardist= stardist.to(device=self.device, dtype=torch.float32)
                objprob = objprob.to(device=self.device, dtype=torch.float32)

                # compute weighted sum of losses
                new_total_loss, new_loss_overlap, new_loss_stardist, new_loss_objprob, new_prec_overlap, new_prec_stardist, new_prec_objprob = self.mtl(
                    images, overlap, stardist, objprob
                    )
                
                # accumulate loss across iterations
                loss_total += new_total_loss
                loss_overlap += new_loss_overlap
                loss_stardist += new_loss_stardist
                loss_objprob += new_loss_objprob
                prec_overlap += new_prec_overlap
                prec_stardist += new_prec_stardist
                prec_objprob += new_prec_objprob

            # compute average loss 
            loss_total = loss_total / num_iter_test
            loss_overlap = loss_overlap / num_iter_test
            loss_stardist = loss_stardist / num_iter_test
            loss_objprob = loss_objprob / num_iter_test
            prec_overlap = prec_overlap / num_iter_test
            prec_stardidt = prec_stardist / num_iter_test
            prec_objprob = prec_objprob / num_iter_test

            # write losses to tensorboard
            self.writer.add_scalar('Test set total loss', loss_total, step)
            self.writer.add_scalar('Test set overlap loss', loss_overlap, step)
            self.writer.add_scalar('Test set stardistances loss', loss_stardist, step)
            self.writer.add_scalar('Test set object probabilities loss', loss_objprob, step)
            self.writer.add_scalar("Test set overlap precision", prec_overlap, step)
            self.writer.add_scalar("Test set stardistances precision", prec_stardist, step)
            self.writer.add_scalar("Test set object probabilities precision", prec_objprob, step)