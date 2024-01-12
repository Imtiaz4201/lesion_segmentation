# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
input_dir = 'dataset/test'
output_dir = 'dataset/output'

# you are allowed to import other Python packages above
##########################


def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    def contrast_stretching(image):
        min_intensity = np.min(image)
        max_intensity = np.max(image)
        stretched_image = ((image - min_intensity) /
                           (max_intensity - min_intensity)) * 255
        stretched_image = stretched_image.astype(np.uint8)
        return stretched_image

    def extend_roi(roi, extend_pixels_x=10, extend_pixels_y=10, image_shape=None):
        x, y, w, h = roi[0][0], roi[0][1], roi[1][0], roi[1][1]

        # Check if extension exceeds image boundaries
        if x - extend_pixels_x < 0:
            extend_pixels_x = x
        if y - extend_pixels_y < 0:
            extend_pixels_y = y
        if x + w + extend_pixels_x > image_shape[1]:
            extend_pixels_x = image_shape[1] - (x + w)
        if y + h + extend_pixels_y > image_shape[0]:
            extend_pixels_y = image_shape[0] - (y + h)

        # Extend the ROI in each direction
        extended_roi = [[x - extend_pixels_x, y - extend_pixels_y],
                        [w + 2 * extend_pixels_x, h + 2 * extend_pixels_y]]

        return extended_roi

    def extend_roi_percentage(roi, Wpercentage=10, Hpercentage=10, image_shape=None):
        x, y, w, h = roi[0][0], roi[0][1], roi[1][0], roi[1][1]

        # Calculate extension values based on percentage of width and height
        extension_x = int(w * Wpercentage / 100)
        extension_y = int(h * Hpercentage / 100)

        # Check if extension exceeds image boundaries
        if x - extension_x < 0:
            extension_x = x
        if y - extension_y < 0:
            extension_y = y
        if x + w + extension_x > image_shape[1]:
            extension_x = image_shape[1] - (x + w)
        if y + h + extension_y > image_shape[0]:
            extension_y = image_shape[0] - (y + h)

        # Extend the ROI in each direction
        extended_roi = [[x - extension_x, y - extension_y],
                        [w + 2 * extension_x, h + 2 * extension_y]]

        return extended_roi

    def get_roi(img):
        """
        to get the lesion region and return the biggest ROI coordinates
        """
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # RGB to LAB
        saturation_channel = lab_image[:, :, 1]  # take L
        saturation_channel = contrast_stretching(
            saturation_channel)  # adjust brightness
        _, trsh = cv2.threshold(saturation_channel, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # get target

        # eliminate coners boxes , get  largest box
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            trsh, connectivity=4)
        # Define a threshold for the aspect ratio to filter out elongated structures
        aspect_ratio_threshold = 10.0  # Adjust as needed
        cdt = (
            trsh.shape[0] + trsh.shape[1])//10  # Adjust as needed /corner distance threshold

        # Keep track of the biggest ROI
        biggest_roi = None
        max_roi_area = 0

        # Iterate through each connected component and filter based on aspect ratio
        for l in range(1, num_labels):
            # Get the bounding box and centroid of the connected component
            x, y, w, h, _ = stats[l]
            centroid = centroids[l]
            # Calculate the aspect ratio
            aspect_ratio = w / h if h != 0 else 0
            # if the components not close to the corners
            if aspect_ratio < aspect_ratio_threshold and \
                    centroid[0] > cdt and centroid[0] < trsh.shape[1] - cdt and \
                    centroid[1] > cdt and centroid[1] < trsh.shape[0] - cdt:
                # Check if the current ROI has a larger area than the current biggest ROI
                roi_area = w * h
                if roi_area > max_roi_area:
                    max_roi_area = roi_area
                    biggest_roi = [[x, y], [w, h]]
        # biggest_roi = extend_roi_percentage(biggest_roi, 20,50, saturation_channel.shape)
        biggest_roi = extend_roi(biggest_roi, 60, 40, saturation_channel.shape) #60,40
        return biggest_roi

    def evolve_level_set_function(level_set_function, image, regularization_coefficient, speed_term_coefficient, epsilon, time_step):
        # active contour modles / snakes
        # Calculate the Dirac function
        dirac = (epsilon / np.pi) / (epsilon**2 + level_set_function**2)

        # Calculate the Heaviside function
        heaviside = 0.5 * (1 + (2 / np.pi) *
                           np.arctan(level_set_function / epsilon))

        # Compute gradients of level set function
        gradient_y, gradient_x = np.gradient(level_set_function)
        magnitude_gradient = np.sqrt(gradient_x**2 + gradient_y**2)

        # Normalize gradients
        normal_x = gradient_x / (magnitude_gradient + 0.000001)
        normal_y = gradient_y / (magnitude_gradient + 0.000001)

        # Compute second-order derivatives of normalized gradients
        second_derivative_xx, second_derivative_yy = np.gradient(normal_x)
        second_derivative_xy, second_derivative_yx = np.gradient(normal_y)

        # Calculate curvature term
        curvature_term = second_derivative_xx + second_derivative_yy

        # Compute length term
        length_term = speed_term_coefficient * dirac * curvature_term

        # Laplacian of the level set function
        laplacian = cv2.Laplacian(level_set_function, -1)

        # Penalty term
        penalty_term = regularization_coefficient * \
            (laplacian - curvature_term)

        # Calculate average intensity inside and outside the contour
        average_intensity_inside = np.sum(
            heaviside * image) / np.sum(heaviside)
        average_intensity_outside = np.sum(
            (1 - heaviside) * image) / np.sum(1 - heaviside)

        # Chan-Vese term
        chan_vese_term = dirac * \
            (-1 * (image - average_intensity_inside)**2 +
             1 * (image - average_intensity_outside)**2)

        # Update the level set function
        level_set_function = level_set_function + time_step * \
            (length_term + penalty_term + chan_vese_term)

        return level_set_function
    
    def hair_remove(image): 
        # remove hair from skin
        grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # smoothing
        kernel = cv2.getStructuringElement(1,(75,75)) 
        bhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel) 
        _,threshold = cv2.threshold(bhat,10,255,cv2.THRESH_BINARY)
        # inpaint with original image and threshold image
        inpaint = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
        inpaint = cv2.medianBlur(inpaint,5)
        return inpaint

    def processed_image(img):
        # bgr to gary
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # smoothing image
        image = cv2.medianBlur(image, 11)
        # Get ROI coordinates
        roi_coordinates = get_roi(img)
        x, y = roi_coordinates[0]
        w, h = roi_coordinates[1]
        # Extract the region inside the bounding box
        lesion_roi = image[y:y+h, x:x+w, :]
        lesion_roi = hair_remove(lesion_roi)

        # convert ROI extracted BGR space to GRAY
        #lesion_roi = cv2.GaussianBlur(lesion_roi, (3, 3), 0)
        lesion_roi = cv2.cvtColor(lesion_roi, cv2.COLOR_BGR2GRAY)

        #  equaliz brightness
        t = cv2.equalizeHist(lesion_roi)

        # Initialization of the level/ segmenting the lesion using active contour models technique
        initial_level = np.ones_like(t, dtype=np.float64)
        initial_level[30:80, 30:80] = -1
        initial_level = -initial_level
        # Evolve the level set function over iterations
        for _ in range(1, 10):
            initial_level = evolve_level_set_function(
                initial_level, t, 1, 0.003 * 255 * 255, 1, 0.1)
            # contours, _ = cv2.findContours(
            #     np.uint8((initial_level > 0)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Set positive values in the level set function to 1
        initial_level[initial_level > 0] = 1

        _, trsh = cv2.threshold(initial_level.astype(np.uint8), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # get target

        blank = np.zeros((image.shape[0], image.shape[1]))
        blank[y:y+h, x:x+w] = trsh
        blank = blank.astype(np.uint8)

        saturation_dilate = cv2.erode(
            blank, np.ones((3, 3), np.uint8), iterations=0)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            saturation_dilate, connectivity=4)
        # find the index (label) of the connected component with the largest area
        largest_centroid = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_centroid_mask = (labels == largest_centroid).astype(
            np.uint8) # create a binary mask of largest_centroid

        outPut_image = cv2.dilate(largest_centroid_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)), iterations=2)
        return outPut_image

    pImg = processed_image(img)
    return pImg
    # END OF YOUR CODE
    #########################################################################
