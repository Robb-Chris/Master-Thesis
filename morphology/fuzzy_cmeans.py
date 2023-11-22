from skimage import measure
import skfuzzy as fuzz
import numpy as np


def fuzzy_cmeans(image, mask, seed_points, param):
    """
    image: image to be segmented
    mask: mask of the image
    seed_points: list of seed points
    param: parameter for the segmentation
    
    """

    # Looping through all seed points to evaluate the segmentation
    segment_mask = np.zeros_like(mask)

    for seed in seed_points:
        # Flatten image
        shape = image.shape
        image_flattened = image.reshape(-1)

        # FCM
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            image_flattened.reshape(1, -1),
            param.get("clusters", 2), # Number of clusters
            param.get("fuzziness", 1.5), # Fuzziness parameter
            error=param.get("error", 0.0001),
            maxiter=param.get("maxiter", 1000),
            init=None)

        # cluster membership
        labels = np.argmax(u, axis=0)
        segmented_image = labels.reshape(shape)
        labels = measure.label(segmented_image)
        properties = measure.regionprops(labels)

        # label of the segment at the seed point
        seed_label = labels[seed]
        # Create a mask for the segment
        mask = labels == seed_label

        segment_mask += mask

        # ------------ DEBUG ------------
        # plt.figure(figsize=(8, 8))

        # plt.subplot(2, 2, 1)
        # plt.imshow(crop_area(image), cmap="gray"), plt.title("Original Image"), plt.axis("off")
        # plt.scatter(adjusted_seed[0], adjusted_seed[1], color="red", s=2)

        # plt.subplot(2, 2, 2)
        # plt.imshow(segmented_image, cmap="gray"), plt.title("Segmented Image"), plt.axis("off")
        # for prop in properties:
        #     centroid = prop.centroid
        #     plt.text(centroid[1], centroid[0], str(prop.label), color="red", fontsize=12)

        # plt.subplot(2, 2, 3)
        # plt.imshow(segment_mask, cmap="gray"), plt.title("Segment Mask"), plt.axis("off")

        # plt.subplot(2, 2, 4)
        # plt.imshow(crop_area(image), cmap="gray")
        # plt.imshow(segment_mask, cmap="gray", alpha=0.5), plt.title(f"Segment containing seed point {seed_points[0]}"), plt.axis("off")
        # plt.scatter(adjusted_seed[0], adjusted_seed[1], color="red", s=2)

        # plt.show()
        # ------------ DEBUG ------------

    
    segment_mask = np.where(segment_mask > 0, 1, 0)
    return segment_mask