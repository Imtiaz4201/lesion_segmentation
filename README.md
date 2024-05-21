The task involves segmenting lesions within medical images, which is an important step in
medical image analysis. The importance of accurate lesion segmentation for various medical
applications such as disease diagnosis, treatment planning, and monitoring motivates this task.
Accurate segmentation assists healthcare providers in identifying and analyzing unhealthy
regions within medical images, resulting in more informed medical decisions.Lesion
segmentation automation is critical to increasing workflow efficiency in medicine, decreasing
manual labor, and possibly increasing diagnostic precision. Furthermore, automated
segmentation can make it easier to extract quantitative data from medical images, which can
help with clinical studies and research.The suggested approach aims to ease the difficulties
related to lesion segmentation, offering medical practitioners a possible instrument to improve
their diagnostic proficiency. The method is useful in a variety of medical specialties and can be
applied to different types of medical imaging.
<br>
<h2>Description of Methods Figure</h2>
<p align="center">
  <img src="https://github.com/Imtiaz4201/lesion_segmentation/blob/main/dataset/report%20Image%5B3%5D.jpg" width="350" title="hover text">
</p>
<br>
The preprocessing phase begins with a conversion of the input image from the BGR to RGB
color space, facilitating consistent color representation. Subsequently, a median filtering
operation is applied to the image for smoothing, reducing noise, and enhancing overall quality.
The identification of the Region of Interest (ROI) involves a series of steps. First, the image is
converted to the LAB color space, and the saturation channel is extracted. To enhance visibility,
a contrast-stretching technique is employed on the saturation channel. Otsu's method is then
utilized for optimal thresholding, leading to the identification of the largest ROI. The aspect ratio
and distance from the image edges are considered during this process. Furthermore, the
identified ROI is extended to encompass surrounding areas, ensuring a comprehensive
representation.
In the lesion segmentation phase, the process initiates with the extraction of the lesion region
using the ROI coordinates acquired from the Region of Interest (ROI) function. Subsequently,
the cropped image undergoes a series of preprocessing steps aimed at enhancing its quality.
These steps include hair removal , denoising, conversion from rgb to gray space. To address
variations in lighting conditions, histogram equalization is applied, enhancing the overall contrast
of the image.The next step involves employing active contour models, a segmentation
technique that utilizes level sets, to delineate the boundaries of the lesion within the
preprocessed image. This technique aids in achieving a more refined representation of the
lesion region.Following the contour-based segmentation, thresholding is applied to create a
binary representation of the lesion. The segmented region is then reshaped to its original size,
ensuring consistency with the input image dimensions. To address potential artifacts and
fine-tune the segmentation, morphological operations are performed, contributing to the overall
cleanliness of the segmented lesion.Continuing the refinement process, connected components
analysis is utilized to eliminate small, isolated regions in the binary lesion representation.
Additionally, Canny edge detection and contour detection are employed to fill up any hollow
areas within the binary lesion, further enhancing the completeness and accuracy of the
segmentation.
<h2>Results</h2>
● Adapted Rand Error: 6% <br>
● Precision: 94% <br>
● Recall: 94% <br>
● IoU: 88% 
