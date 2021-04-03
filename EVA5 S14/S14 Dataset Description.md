During the 13th Assignment, we were provided the dataset which was contributed by the EVA 5 group. The dataset had 4 classes, namely Hardhat, Vest, Boots, and Masks. It was annotated for the Yolo V3 algorithm with bounding boxes. 

For the 14th Assignment, our task was to create three sets of images, the first being the Depth Map using MIDAS, the next was Surface Planes using PlaneRCNN along with Bounding boxes from assignment 13. We have created a folder with 3 subfolders with the above-mentioned image dataset.

MIDAS Depth Map: Monocular Depth Estimation 
We have used the MIDAS model to generate a depth map. The MIDAS technique was published in the research paper titled “Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer”  Here we do Monocular depth estimation from a single rgb image. The output of the algorithm is a single image file with a depth map differentiated by the intensity of the different areas of the map. This type of data can be used for multiple applications including Augmented Reality, Robotics, Autonomous Systems and Self- Driving Cars.


PlaneRCNN Surface Planes: We have used PlaneRCNN algorithm to make surface planes along with depth maps for an image. Here we only use surface planes and drop the depth maps.  PlaneRCNN was published in the research paper “PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image” by NVIDIA Research. This algorithm detects and reconstructs piecewise plane surfaces using a variant of MaskRCNN. As input, we need just a single RGB image. For every single input, it provides 9 output files including DEPTH Maps, 3Dplanes, and planer regions.  Different regions are annotated using different colored masks.


Bounding Boxes: The dataset has 4 classes and it was annotated using an annotator tool that provides us with coordinates of bounding boxes along with their dimensions. Folder consists of 
i). classes.txt => Contains names of the four classes
ii). custom.names
iii). custom. data => Number of classes and path of files containing names of train, validation images and path to custom.name file
iv) test.shapes
v)  test.txt => Description of files to  be used for testing.
vi) train.shapes
vii)train.txt
viii) images(Folder) => Contains images for bounding box 
ix) labels(Folder) => Contains txt file for every image - has coordinates related to bounding box (x, y, h, w)
