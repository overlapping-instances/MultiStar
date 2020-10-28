import numpy as np
from PIL import Image, ImageDraw

def get_polygon_coordinates(x, y, stardist):
    """Generate a polygon from star distances and determine the coordinates of its vertices.
    
    Parameters:
    x -- x coordinate of polygon center
    y -- y coordinate of polygon center
    stardist -- array of shape (num_rays, H, W) with the polygon distances

    Returns: list of coordinate tuples.
    """

    angles = np.linspace(0, 2 * np.pi, stardist.shape[0], endpoint=False)
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)

    coordinates_x = x + stardist[:, y, x] * cos_angles
    coordinates_y = y + stardist[:, y, x] * sin_angles

    coordinates = tuple(map(tuple, np.stack((coordinates_x, coordinates_y), axis=1)))
    
    return coordinates


def compute_IoU(mask1, mask2, overlap):
    """Compute the intersection over union of two masks, weighted with (1 - overlap).

    Parameters:
    mask1 -- binary array
    mask2 -- binary array of same shape as mask1
    overlap -- array of same shape as mask1 and mask2

    Returns: IoU of mask1 and mask2 with weighted intersection
    """

    intersection = np.logical_and(mask1, mask2) * (1 - overlap)
    union = np.logical_or(mask1, mask2)

    return intersection.sum() / (np.count_nonzero(union) + 1e-6)

    
def sample_positions(objprob, num_samples, overlap, min_objprob):
    """Sample pixel positions with probabilities proportional to the object probabilities.

    Pixels are sampled with reduced probability in overlap regions and not where the object probability is lower than min_objprob.
    Parameters:
    objprob -- array of shape (H, W) with object probabilities
    num_samples -- number of pixel positions to be sampled
    overlap -- array of shape (H, W) with overlap
    min_objprob -- minimum required object probability for sampling a pixel

    Returns: array of shape (num_proposals, 2) containing x/y coordinates of sampled
    points, array of shape (H, W) containing a mask for the sampled points.
    """

    objprob = objprob.copy()

    image_height = objprob.shape[0]
    image_width = objprob.shape[1]

    polygon_pixel_coordinates = np.zeros((num_samples, 2), dtype='int32')
    samples_mask = np.zeros_like(objprob, dtype='bool')

    # make it less likely to sample in the overlap
    objprob = objprob * (1 - overlap)

    # don't sample pixels with too low object probability
    objprob = objprob * (objprob > min_objprob)

    # normalize to make applicable as probability distribution
    objprob = objprob / objprob.sum()

    objprob = objprob.flatten()
    cum_objprob = np.cumsum(objprob)

    # map uniform samples to pixels
    for i, z in enumerate(np.random.uniform(size=num_samples)):
        diff = cum_objprob - z

        # consider only cases where x is greater than objprobs
        # therefore mask away zero and negative difference
        mask = np.ma.less_equal(diff, 0)
        masked_diff = np.ma.masked_array(diff, mask)

        # position in 1D-array where x is closest but greater than array value
        min_pos = masked_diff.argmin()

        # retrieve pixel coordinates
        x = min_pos % image_width
        y = min_pos // image_width

        samples_mask[y, x] = True

        polygon_pixel_coordinates[i, 0] = x
        polygon_pixel_coordinates[i, 1] = y
        
    return polygon_pixel_coordinates, samples_mask


def generate_polygon_masks(polygon_pixel_coordinates, stardist):
    """Generate binary masks of polygons at the specified coordinates with given star distances.
    
    Parameters:
    polygon_pixel_coordinates -- array of shape (num_proposals, 2) with x/y coordinates of polygon positions
    stardist -- array of shape (num_rays, height, width)

    Returns: binary array with polygon mask
    """

    polygons = np.empty((polygon_pixel_coordinates.shape[0], stardist.shape[1], stardist.shape[2]), dtype='bool')

    for i in range(polygon_pixel_coordinates.shape[0]):
        # get vertex coordinates
        coordinates = get_polygon_coordinates(
            polygon_pixel_coordinates[i, 0],
            polygon_pixel_coordinates[i, 1],
            stardist
            )

        # generate mask
        img = Image.new('L', (stardist.shape[1], stardist.shape[2]), 0)
        ImageDraw.Draw(img).polygon(coordinates, outline=1, fill=1)
        polygons[i] = np.array(img)
    
    return polygons


def nms(overlap, stardist, objprob, num_proposals, iou_thres, min_objprob):
    """Generate proposals and perform non-maximum suppression.
    
    Parameters:
    overlap -- array of shape (1, H, W) with the predicted overlap (values between 0 and 1)
    stardist -- array of shape (num_rays, H, W) with the ray lengths
    objprob -- array of shape (1, H, W) with the object probabilities (values between 0 and 1)
    num_proposals -- number of proposals to generate
    iou_thres -- maximum allowed intersection over union for polygons to coexist (not relevant for overlap regions)
    min_objprob -- minimum required object probability to sample from pixel
    
    Returns: array of shape (num_polygons, H, W) with the final proposals
    """

    # sample polygon positions based on object probabilities
    polygon_pixel_coordinates, samples_mask = sample_positions(objprob, num_proposals, overlap, min_objprob)

    # generate a masks for the polygons centered at polygon_pixel_coordinates
    polygons = generate_polygon_masks(polygon_pixel_coordinates, stardist)

    # get an array with the object probabilities of the sampled pixels
    objprob = objprob.flatten()[samples_mask.flatten()]
    
    # sort the polygons wrt to their object probabilities in descending order
    sorted_polygon_indices = np.flip(np.argsort(objprob))


    accepted_polygons = []
    for i in sorted_polygon_indices:
        for j in accepted_polygons:
            score = compute_IoU(polygons[i], polygons[j], overlap)
            if score > iou_thres:
                break
        else:
            accepted_polygons.append(i)

    return polygons[accepted_polygons].astype('bool'), polygon_pixel_coordinates[accepted_polygons]


def plot_one_polygon(x, y, image, stardist):
    """Plot the predicted polygon at coordinates x, y.

    Parameters:
    x -- x coordinate of polygon center
    y -- y coordinate of polygon center
    image -- original image
    stardist -- array with stardistances, last two dimensions agree with image dimensions
    """

    image_height = stardist.shape[1]
    image_width = stardist.shape[2]
    num_rays = stardist.shape[0]

    # compute the ray angles and their sin/cos values
    angles, sin_angles, cos_angles = get_sin_cos_angles(num_rays)

    # get coordinates of polygon corners
    coordinates = get_polygon_coordinates(x, y, stardist, angles, sin_angles, cos_angles)

    binary_polygon = polygon2mask((image_height, image_width), coordinates).T

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 15))

    axs[0].set_title('Original image')
    axs[1].set_title('Polygon at x=%s, y=%s' %(x, y))

    axs[0].imshow(image, cmap='gray')
    axs[1].imshow(binary_polygon, vmin=0, vmax=1.0, cmap='gray')

    plt.show()