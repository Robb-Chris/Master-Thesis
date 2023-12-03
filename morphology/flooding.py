from skimage import segmentation, measure
import matplotlib.pyplot as plt
import numpy as np


def flooding(image, mask, seed_points, param):
    """
    image: image to be segmented
    mask: mask of the image
    seed_points: list of seed points
    param: parameter for the segmentation
    
    """

    # Looping through all seed points to evaluate the segmentation
    segment_mask = np.zeros_like(mask)

    for seed in seed_points:
        label = segmentation.flood(image, seed, connectivity=param.get("connectivity", 2), tolerance=param.get("tolerance", 30))

        segment_mask += label

        # ------------ DEBUG ------------
        # plt.figure(figsize=(12, 12))

        # plt.subplot(2, 2, 1)
        # plt.imshow(crop_area(image) , cmap="gray"), plt.title("Original"), plt.axis("off")

        # plt.subplot(2, 2, 2)
        # plt.imshow(crop_area(image), cmap="gray")
        # plt.imshow(crop_area(mask), cmap="nipy_spectral", alpha=0.5)
        # # for (y, x) in seed_points:
        # #     plt.scatter(x, y, color="red", s=1, alpha=0.2)
        # plt.title("Original Mask on Image"), plt.axis("off")

        # plt.subplot(2, 2, 3)
        # plt.imshow(crop_area(labels) , cmap="gray"), plt.title("Watershed Mask"), plt.axis("off")

        # plt.subplot(2, 2, 4)
        # plt.imshow(crop_area(image), cmap="gray")
        # plt.imshow(crop_area(labels), cmap="nipy_spectral", alpha=0.5)
        # # for (y, x) in seed_points:
        # #     plt.scatter(x, y, color="red", s=1, alpha=0.2)
        # plt.title("Watershed Mask on Image"), plt.axis("off")

        # plt.show()
        # ------------ DEBUG ------------

    
    segment_mask = np.where(segment_mask > 0, 1, 0)
    return segment_mask