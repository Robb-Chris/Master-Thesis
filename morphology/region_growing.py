import numpy as np

def region_growing(image, mask, seed_points, param):
    """
    image: image to be segmented
    mask: mask of the image
    seed_points: list of seed points
    param: parameter for the segmentation
    
    """

    # Looping through all seed points to evaluate the segmentation
    segment_mask = np.zeros_like(mask)

    height, width = image.shape
    region = np.zeros_like(image, dtype=np.uint8) 
    points_to_visit = [seed_points]

    # The mean intensity of the region; starting with the seed intensity
    region_mean = float(image[seed_points])
    region[seed_points] = 255

    if param.get("connectivity", 8):
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    else:
        neighbours = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Loop until there are no more points to visit
    while points_to_visit:
        y, x = points_to_visit.pop(0)

        # Check the neighbors of the current point
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx

            # Skip out-of-bounds or already added points
            if nx < 0 or ny < 0 or nx >= height or ny >= width or region[ny, nx] == 255:
                continue

            # Calculate intensity difference
            intensity_diff = abs(int(image[ny, nx]) - region_mean)

            if intensity_diff < param.get("threshold", 22):
                region[ny, nx] = 255
                points_to_visit.append((ny, nx))
                # Recalculate the mean intensity of the region
                region_mean = ((region_mean * np.sum(region == 255)) + int(image[ny, nx])) / (np.sum(region == 255) + 1)
        
        segment_mask += region

    
    segment_mask = np.where(segment_mask > 0, 1, 0)
    return segment_mask