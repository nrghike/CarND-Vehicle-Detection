# Self-Driving Car Engineer Nanodegree

# Project: Vehicle Detection

# Submitted By: Ninad Ghike


##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.


To run the code, grab the training data from 
https://github.com/udacity/CarND-Vehicle-Detection and step through the vehicle_detection.ipynb.


[//]: # (Image References)
[image1]: ./output_images/Dataset_visualization.png
[image2]: ./output_images/Hog_feature_Visualization.png
[image3]: ./output_images/Heat_Map.png
[image4]: ./output_images/Feature_Vector_Normalization.png
[image5]: ./output_images/Sliding_window_Search.png
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

```python
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]



####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Increasing the number of histogram bins and orientations increased accuracy but took more time to compute. Larger cells were faster but less accurate. Finally the following combination was a balance between the computation and accuracy:

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The features of the cars and no-cars were stacked and accordingly even the corresponding labels were stacked.

One of the most important operation performed on the data was that of normalization. Every type of feature has output of varying either from 0-1 or from 0-255. It is important that all of them are normalized. Also, the same normalization operation must be performed on the test feature vector.

The data was split between 80% training and 20% testing. This was important to gause the accuracy of the model.

Finally, a Linear SVM was used to train.

```python
# Prepare and normalize data

# Define the features.
X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)                        

X_scaler = StandardScaler()
scaled_X = X_scaler.fit_transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

# Visualize feature vectors for vehicle vs non-vehicle
VisualizeFeatures(X[8], scaled_X[8], X[-36], scaled_X[-36])
# VisualizeFeatures(X[74], scaled_X[74], X[-1255], scaled_X[-1255])
# VisualizeFeatures(X[200], scaled_X[200], X[-332], scaled_X[-332])
print('X shape: ', X.shape)
print('Y shape: ', y.shape)

# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=123)

svc = LinearSVC()
start_time=time.time()
svc.fit(X_train, y_train)
end_time = time.time()

```


![Alt Text][image4]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I iterated through the hyperparameter settings by using performance on the test images.
I decided to search only the bottom half of the image to cut the computation and false positive
since the camera position is fixed on the car. I used two different sliding window size of
64x64 and 128x128 to be able to detect car of varying sizes.

This is in the ```find_cars``` function

```python
 # Test vehicle detection on test images using two different sliding window sizes.

for img in glob.glob("test_images/*.jpg"):
    image = mpimg.imread(img)
    fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

    annotated_img, bboxes = find_cars(image, np.int(image.shape[0]/2), image.shape[0], 1, svc, X_scaler)
    ax1.imshow(annotated_img)
    ax1.set_title("Sliding window: 64x64", fontsize=24)

    annotated_img, bboxes = find_cars(image, np.int(image.shape[0]/2), image.shape[0], 2, svc, X_scaler)
    ax2.imshow(annotated_img)
    ax2.set_title("Sliding window: 128x128", fontsize=24)
```


![Alt Text][image5]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned
color and histograms of color in the feature vector, which provided a nice
result.  I also used heatmap to combine the output from the two different sliding window
scales to remove false positive for each frame. Here are some example images:

![Alt Text][image3]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./output_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video, keeping results
from the last 7 frames.  From the positive detections I created a heatmap and then
thresholded that map (to at least have 4 frames with the boxes) to identify vehicle positions.  I then used
`scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

```python
# Visualize the effect of using heatmap to combine detection
# from two different sliding window size.

for img in glob.glob("test_images/*.jpg"):
    image = mpimg.imread(img)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    fig = plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    
    # 64 x 64 sliding window.
    annotated_img, bboxes = find_cars(image, np.int(image.shape[0]/2), image.shape[0], 1, svc, X_scaler)
    heat = add_heat(heat, bboxes)
    ax1.imshow(annotated_img)
    ax1.set_title("Sliding window: 64x64", fontsize=24)

    # 128 x 128 sliding window.
    annotated_img, bboxes = find_cars(image, np.int(image.shape[0]/2), image.shape[0], 2, svc, X_scaler)
    heat = add_heat(heat, bboxes)
    ax2.imshow(annotated_img)
    ax2.set_title("Sliding window: 128x128", fontsize=24)
    
    # Aggregate the heatmap.
    heat = apply_threshold(heat, 1)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    combined_img, bboxes = draw_labeled_bboxes(np.copy(image), labels)
    ax3.imshow(combined_img)
    ax3.set_title("Combined by heatmap", fontsize=24)
```

---


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If we can spend more time, there are a couple of things to improve the pipeline.

Multi-scale. Right now, I only used 2 different scales but this cause problem when the car enters
the scene and when the car is very far from the view. Adding more scales will help but this cause the pipeline
to be a lot slower. One way to make it faster is to use different scale at different location of the image
(eg. bigger scale when it is at the bottom of the image).

Close cars. Right now, when two cars are very close to each other, the pipeline only detect one bounding box.
This is not ideal, and we should consider how to split the two boxes when computing the heat map and using
the labels function. This might
involve using different threshold and different # of frames.

