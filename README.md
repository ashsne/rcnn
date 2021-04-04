# RCNN for object detection

Object detection is technique in computer vision and image processing to identify and locate objects in an image or a video frame. The state-of-the art methods can be categorized into two types: one-stage methods and tow-stage methods. One-stage methods prioritize inference speed eg.g YOLO, SSD and RetinaNet. Two stage methods prioritize detection accuracy e.g. Faster R-CNN, Mask RCNN and Cascade RCNN.

The current state-of-the-art two-stage methods, are mostly based on the pioneering proposal from Girshick et al in 2014 called “Rich feature hierarchies for accurate object detection and semantic segmentation”. The method which uses region based convolutional neural networks is dubbed as RCNN.  

<p align="center">
  <img width="460" src="https://user-images.githubusercontent.com/51709130/113511518-d5bcf500-9568-11eb-9c83-90d6dfabe991.png">
  <br><br>
  <a href="https://arxiv.org/abs/1311.2524">RCNN: Regions with CNN features</a> 
</p>

The current implementation differes from the original work as below:
 - instead of Alexnet, various backbone models can be used eg. VGG16, VGG19, mobilenet etc. 
 - instead of SVMs for each class, the current work proposes an head NN for classification trained with categorical entropy class

### Region proposals 
Region proposals solve the very computation heavy brute-force method of sliding-window approach. The RCNN apporach uses 'selective-search (SS)' for region proposals. A well documented difference between selective search and sliding window approach can be fing [here](https://learnopencv.com/selective-search-for-object-detection-cpp-python/). 

The SS algorithm proposes nearly 2000 different regions for the object detection task.

### CNN
In the next step, each region proposed goes through a CNN to produce a feature vector. The original paper used the well-known architecture of classification at the time, Alexnet. The current implementation is modified to use VGG16.

<p align="center">
  <img width="460" src="https://user-images.githubusercontent.com/51709130/113511870-b4f59f00-956a-11eb-9f81-ee09fe1b38b2.png">
  <br><br>
  <a href="https://medium.com/@selfouly/r-cnn-3a9beddfd55a">CNN</a> 
</p>

### SVM
The created features from the CNN are used for classification task. For this, we use a SVM classificator. We have one SVM for each object class and we use them all. The output of the SVM is a confidence score for the class.

### Non-max supression
Now we have 2000 regions and respective confidence scores for the classes. With a greedy alogorithm, non-max suppresion, appropriate regions are combined. The method rejects a region if it has an IOU overlap with higher scoring selected region.
