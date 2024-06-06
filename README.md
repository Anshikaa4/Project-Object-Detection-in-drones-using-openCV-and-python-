#This is a Object detection in drones Project
<br>
<br>
Drone- An unmanned aerial vehicle (UAV), commonly known as a drone, is an aircraft without any human pilot, crew, or passengers on board.
<br>
Object Detection
You can use a variety of methods and libraries to carry out object detection in drone imagery. Here is a general how-to using a few popular libraries and Python:

1. Get Drone Imagery: Retrieve pictures or videos that the drone's camera has taken. You have the option of using either pre-recorded videos or images, or live video feeds.

2. Select an Object Detection Technique:

i. Pre-trained Models: You can employ models such as SSD (Single Shot MultiBox Detector), YOLO (You Only Look Once), or Faster R-CNN that have already been trained. These models can identify a variety of objects because they were trained on large datasets.
ii. Custom Models: Using deep learning frameworks such as TensorFlow or PyTorch, you may need to train your own model if you have specific objects to detect. A labelled dataset and training procedure are needed for this.

3. Put Object Detection into Action:

In most cases, you can find Python libraries or wrappers to make it simple to use pre-trained models. For instance, you can load pre-trained models like YOLO or SSD using OpenCV's cv2.dnn module.
You'll need to write code to load the model and carry out inference on the drone imagery if you're training your own models.

4. Post processing:

It might be beneficial to carry out post-processing to eliminate false positives or improve the bounding boxes that were found after detection.
Other tasks such as tracking objects across frames to obtain trajectories can also be executed.

5. Visualise the Outcomes:

Lastly, display the object detection results. You have the option to superimpose data on the original image or draw bounding boxes around objects that are detected.



