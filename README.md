**INTRODUCTION DATASET**

The following is a dataset derived from the YouTube Facial Palsy dataset, which was manually annotated using Roboflow. This dataset consists of 1,012 images categorized into three classes: Normal, Medium, and Strong.
Dataset : https://drive.google.com/file/d/1nbtJjyFum9KpQn_3V4SmPcj97_3nl_yK/view?usp=sharing

**CONCLUSION**
Based on the results, it can be concluded that the Detection Transformer (DETR) method successfully detects Facial Paralysis based on three predefined classes, namely Normal, Medium, and Strong. However, there are some misclassifications, especially in the Normal class which is often classified as Medium, as well as some samples from the Strong class which are also predicted as Medium. This indicates an overlap of features between classes that have similar characteristics. The model performance evaluation showed that  DETR with Backbone ResNet-50 performed better than DETR using Backbone ResNet-101. DETR with Backbone ResNet-50 recorded an mAP value of 0.687, which is higher than DETR using ResNet-101 which obtained an mAP of 0.617. This result shows that the model needs better learning to distinguish features between classes, especially for similar classes.
