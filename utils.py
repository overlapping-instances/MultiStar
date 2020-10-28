import numpy as np
from skimage import measure

def plot_contours(masks, bg_image, ax, center_coordinates=None, level=0.5, cmap='gray', linewidth=1):
    """Plot the contours of the cell masks on the image.
    
    Parameters:
    masks -- boolean array of shape (num_cells, h, w)
    bg_image -- (real) image to plot in the background
    ax -- subplot object to plot on
    center_coordinates -- array of shape (num_masks, 2) with the x/y coordinates of the instance centers
    level -- value along which to find contours
    cmap -- color map for background image
    linewidth -- line width of contours
    """
    
    num_cells = masks.shape[0]
    
    ax.imshow(bg_image, cmap=cmap)
    
    for i in range(num_cells):
        contours = measure.find_contours(masks[i], level)
    
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=linewidth)

    if center_coordinates is not None:
        for x in center_coordinates:
            ax.scatter(x[0], x[1], marker='x')





