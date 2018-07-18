# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: output_images/camera_cal_image_corner_detection/calibration2.jpg "Distorted Detected Corners"
[image2]: output_images/camera_cal_undistorted/calibration2.jpg "Final Result Undistorted Image"
[image3]: test_images/test6.jpg "Distorted Test"
[image4]: output_images/test_images_undistorted/test6.jpg "Undistorted Test"
[image5]: output_images/test_images_thresh_binary/test6.jpg "Final Binary"
[image6]: output_images/test_images_thresh_binary/test2.jpg "Final Binary 2"
[image7]: output_images/test_images_thresh_binary/straight_lines1.jpg "Straight Lines Before Warp"
[image8]: output_images/test_images_bird_eye/straight_lines1.jpg "Warped Straight Lines"
[image9]: output_images/test_images_detected_lines/test3.jpg "Detected Lane Lines"
[image10]: output_images/test_images_detected_lines/test2.jpg "Detected Lane Lines 2"
[image11]: output_images/test_images_final/test4.jpg "Plotted Image With Lines"
[image12]: output_images/test_images_final/test1.jpg "Ploted Image With Lines 2"

[video1]: project_video_completed.mp4 "Video"

#### Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/571/view) points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is called from (line 24) in `p4.py` and the function called is located in `functions.py` (lines 9-61). The function starts by reading in a list of chessboard calibration images located in the `camera_cal` folder along with the corresponding inside corner numbers along the x,y axis of the chessboard images (9, 6). Object points (real world points) are setup and initialized for the calibration based on the same parameters passed into the function (9, 6). These points are then setup in a grid like fashion ((0,0), (0,1), (0,2) etc...) to prepare for image calibration.

After reading in an image and converting to gray scale `cv2.findChessboardCorners()` is called to try and detect the corners in the image based on the inside corner counts. If corners are detected, we can then add the detected corners to a image point list as well as the corresponding object points previously set up to a object points list. From here chessboard corners are drawn on to the distorted image, this is useful because we can visually check that corners have been properly detected.

Once corner detection has been visually verified, the detected image points and object points can be used to calibrate a distortion coefficient (line 54 `functions.py`).
Using the calibration returned, a call to `cv2.undistort()` with the distorted image and calibration yields a final undistorted image (line 57 `functions.py`).

After again visually verifying the correction works, this process is repeated and verified for all the calibration images. After verifying all calibration images, its safe to return a camera calibration to use for the video stream in the project pipeline.


Distorted Image Detected Corners | Final Result Undistorted Image
:---:|:---:
![Distorted Detected Corners][image1] | ![Final Result Undistorted Image][image2]

---

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I have provided an example of a distorted test image as well as an image that was undistorted using a computed camera calibration. (line 39 `p4.py`)

Distorted | Undistorted
:--:|:--:
![Distorted][image3] | ![Undistorted][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color/gradient thresholds to generate two binary images that get overlapped to create a final binary image, which gets thresholded again to produce the final binary image.

I chose to focus on an HLS converted image that gets edge detected across the x axis using `cv2.sobel(x)` on only the L channel, this created the first binary image. The second color binary image was produce by thresholding the S channel only. (lines 82-98 `functions.py`)

I chose to focus on the S and L channels only because they seemed to produce the best lane lines that were free of noise under most case scenarios covered in the `test_images` folder.

After these two binary images are produced, I overlapped the binary images which yielded the most complete lane line and again applied a very low value threshold (non inclusive 0) to the combined color binary images (this reduced even more noise from the binary images). At the same time of overlapping and thresholding, the final binary was converted to just black and white pixels to produce the lane lines clearly defined. (lines 100-109 `functions.py`)


Final Binary Example| Final Binary Example 2
:---:|:---:
![Final Binary][image5] | ![Final Binary 2][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
My perspective transform is located on lines 112-151 of `fuctions.py`. (called from line 66 of `p4.py`) The function takes in a binary image and defines some source points that have been manually selected from straight lane line images. The four points defined formed a trapezoid capturing the lane lines of a straight lane.

```python
# Set up 4 points in a trapazoid to transform the lane lines to a "birds eye" view
tl = [590, 450] # top left
tr = [690, 450] # top right
br = [1100, image_size[1]] # bottom right
bl = [200, image_size[1]] # bottom left
```

I plotted these points on a few images during testing to verify that they represented the trapezoidal shape that defined the field of view for a straight lane. Once confident with the points, I defined the following destination points:

```python
# Set the destination points (abtracted from src_points)
offset = 250

dest_points = np.float32([
			[tl[0] - (offset * 1.4), 0],
			[tr[0] + (offset * 1.4), 0],
			[br[0] - (offset // 4), br[1]],
			[bl[0] + (offset // 4), br[1]]])

```

The destination points are a calculation of the source points, the particular formula chosen that defines the points was found through trial and error, however, the base idea was to stretch the top two points outward and extend them to the top of the frame while keeping the lower two points at the bottom of the frame and pinching them in slightly. I choose to use a formula rather then hardcoding values because it was faster to tune the end result through trial and error. After a few tries with that concept in mind I was able to derive the end formula. I verified my warped image on a set of straight lines to make sure I was keeping lane lines proportionate to one another (keeping them parallel like in real world space) as well as making sure that the straight lines were actually standing straight up and down in the warped image. In addition, I found that by tuning the source and destination points and rechecking the result allowed me to remove some of the noise in the binary images from the sides of the road. This proved to be useful in further augmenting the data to allow for a cleaner lane line detection.

Before Warp Tranform | Warp Transform Birds Eye
:---:|:--:
![Straight Lines Before Warp][image7] | ![Straight Lines After Warp][image8]


Throughout the testing process points were rechecked and tuned while calling `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` (lines 142-148 `functions.py`).

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Finding the lane lines and fitting their position with a polynomial was one of the more time consuming and difficult parts of this project.
My lane line detection function is called from line 78 of `p4.py` and the function used is contained in lines 154-254 of `functions.py`.

In a nutshell the function works as follows:

* Plot a histogram of all the "hot" pixels row wise in the warped binary image (white pixels). From there peaks are identified in the data, the peaks correspond (most likely) to areas where lane lines are contained in the warped image or more importantly for our purposes x axis values. (line 166 `functions.py`)

* Next, a new image is created from the original warped image (we write to this new image later on), midpoints of the histogram data, and left/right lane line midpoints are set. A number of search windows and size of these windows is also set for the image based on the max y pixel value (image height). (line 168-198 `functions.py`)

* From here, all "hot" (white pixels) are identified for x and y axis. (line 183-186 `functions.py`)

* At this point the function starts its search with the defined windows at the base of the image's (max(y)) histogram detected left and right peaks (specifically the mean "hot" x value for each line) as these are most likely the base of the lane lines and the best starting points. From there the next window starts searching for more pixels above the last window centered at the previously detected windows mean "hot" pixel position. (line 202-228 `functions.py`)

Another key part of the function is that when each window centers itself on these lane line pixels, the positions of x mean at various points (bottom to top of image y values) is recorded and added to left and right line lists respectively. This is important because we are later using these mean "hot" x values (ideally centers of lane lines) to fit a 2nd degree polynomial (3 coefficients). From there the computed coefficients are ultimately plugged back into the quadratic function *(y = ax2 + bx + c)* to calculate x points based on the previously detected points.
These new generated points have been generated using coefficients with the lowest error in respect to the original detected points from the binary image. This helps smooth out some irregularities as well as giving the ability to project points if not all of the lane line was detected or present in the binary image. (lines 230-251 `functions.py`)




Detected Lines | Detected Lines 2
:---:|:---:
![Detected Lane Lines][image9] | ![Detected Lane Lines][image10]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curve function is located in `functions.py` lines 257-292. It is called from the main script `p4.py` on line 100. Previously calculated lane points are fitted to another polynomial and converted to real world space using a meter to pixel value.

```python
# Convert pixel space curvature to object space curvature
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Fit new polynomials to x,y in world space
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
```

With a new set of coefficients in real world space, they are then plugged into another formula ([radius of curvature](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)) which yields the curve of the points in real world space as meters.

```python
# Calculate the new radii of curvature in meters
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

To calculate vehicle center in regard to the lanes. The true center of the vehicle is treated as the midpoint of the image. The center of the lane lines are calculated by taking the max(y) x value for left/right lines, adding them together, and dividing by two. This produces a center of lane pixel value that can be subtracted from the midpoint to calculate the offset from center. Positive offset is right of center, negative offset is left of center.

```python
# Find the center of the car in between the lane lines in a x pixel value, calculate an offset from true center (center of image)
# Convert to meters
lane_line_center = (left_fitx[-1] + right_fitx[-1]) / 2 # get x cordinate at bottom of image closest to vehicle
vehicle_center_offset = round(((lane_lines_image.shape[1] / 2) - lane_line_center) * xm_per_pix, 2)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 355-394 of `functions.py`.  Here is an example of my result on a test image:

Plotted Image With Lines | Plotted Image With Lines 2
:---:|:---:
![Plotted Image With Lines][image11] | ![Plotted Image With Lines 2][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](project_video_completed.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue I faced in my implementation of this project was binary image noise in the concrete areas of the video. Because of the change in color from asphalt to concrete it severely impinged my sobel x edge detection as well as my color thresholding. My initial implementation could successfully detected yellow and white lines against a black road very clearly, however, everything fell apart on the concrete sections of the road. The simplest solution that I came up with to clean up the extra noise was to apply another low value threshold mask to the already thresholded binary generated image. This technique was quick, simple, and effectively cleaned up the catastrophic failures I was seeing.
To further refine my solution I begin to implement and experiment with a line class (line 295-352 `functions.py`) that performed some sanity checks and averaging. However, do to the term time constraints I did not get the chance to really dial in the averaging and sanity checks to yield a totally perfect solution.

Lastly, I want to say this pipeline is by no means a generalize solution for all roads. The problem with this implementation is that it accounts for more case scenarios then the P1 solution (IE: Shadows, Concrete, Curves in the road, etc), but there are still more variables out there in the real world. For example, if this video was taken on a cloudy or rainy day I would probably just assume that this implementation is completely useless because the gradient and color thresholds would fall apart and these are the heart of actually finding the lane lines.

If I had to make this more robust and had more time, I think the key take away from this project is that gathering more insightful information is key as well as deciding if that information is good information or not. I only utilized 2 color channels from HLS to detect the lines, but there is definitely more information available from the H channel as well as RGB and HSV channels. You could potentially get useful insights about the world around the vehicle from all of these color spaces. The combinations of thresholding and layering binary images is endless. In addition, adding in some kind of functionality to average the points and reject outlier pixels based on previous data could help smooth out rough sections of road with things like tar splatter, missing lane lines, old patches of lane lines etc. Having more good data and rejecting bad data will always help and provide the best possible answer to the problem.
