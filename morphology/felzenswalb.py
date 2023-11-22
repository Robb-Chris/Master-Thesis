from skimage import segmentation, measure
import matplotlib.pyplot as plt
import numpy as np


def felzenswalb(image, mask, seed_points, param):
    """
    image: image to be segmented
    mask: mask of the image
    seed_points: list of seed points
    param: parameter for the segmentation
    
    """

    # Looping through all seed points to evaluate the segmentation
    segment_mask = np.zeros_like(mask)

    for seed in seed_points:
        # Applying Felzenszwalb
        segments_fz = segmentation.felzenszwalb(image, scale=param.get("scale", 100), sigma=param.get("sigma", 0.4), min_size=param.get("min:_size", 4))

        # Creating Boundaries
        label = segmentation.mark_boundaries(image, segments_fz, (0, 0, 0),mode="thick")

        # Find the unique segments and their centroids
        regions = measure.regionprops(segments_fz)
        centroids = [region.centroid for region in regions]

        # Selecting a specific segment
        selected_segment_id = segments_fz[seed]
        selected_segment = np.where(segments_fz == selected_segment_id, 1, 0)

        segment_mask += selected_segment

        # ------------ DEBUG ------------
        # plt.figure(figsize=(8, 8))

        # plt.subplot(2, 2, 1)
        # plt.imshow(crop_area(image), cmap="gray"), plt.title("Original Image"), plt.axis("off")
        # plt.scatter(adjusted_seed[0], adjusted_seed[1], color="red", s=2)

        # plt.subplot(2, 2, 2)
        # plt.imshow(label, cmap="gray"), plt.title("Segmented Image"), plt.axis("off")
        # for reg in regions:
        #     centroid = reg.centroid
        #     plt.text(centroid[1], centroid[0], str(reg.label), color="red", fontsize=12)

        # plt.subplot(2, 2, 3)
        # plt.imshow(selected_segment, cmap="gray"), plt.title("Segment Mask"), plt.axis("off")

        # plt.subplot(2, 2, 4)
        # plt.imshow(crop_area(image), cmap="gray")
        # plt.imshow(selected_segment, cmap="gray", alpha=0.5), plt.title(f"Segment containing seed point {seed_points[0]}"), plt.axis("off")
        # plt.scatter(adjusted_seed[0], adjusted_seed[1], color="red", s=2)

        # plt.show()
        # ------------ DEBUG ------------

    
    segment_mask = np.where(segment_mask > 0, 1, 0)
    return segment_mask