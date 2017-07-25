# Udacity CARND Project 4 Advanced Lane Lines
# Main script that contains the project rubric pipeline
import functions
from functions import Line
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML


"""
Camera Calibration:
Compute the camera calibration matrix and distortion coefficients with some
calibration chessboard images.
"""

# Calculate the distortion coefficients for the camera being used
# Default parameters have already been set for the calibration images provided, see functions.py
# Reference camera_cal_image_corner_detection vs camera_cal_undistorted images to visually inspect
# calibration image correction consistency
mtx, dist = functions.calibrate_camera()

def process_image(img, mtx=mtx, dist=dist):

	"""
	This is the project pipline to process lane lines from images or video streams.

	*START Pipline*

	Pipeline (test images):
	Undistort test images.
	"""

	# Undistort image, add it to the list of undistorted images
	# and save a copy to "output_images\\test_images_undistorted" for reference
	destination = cv2.undistort(img, mtx, dist, None, mtx)

	# # Uncomment for test image processing
	# plt.imsave('output_images\\test_images_undistorted\\' + image_name, destination)


	"""
	Pipeline (test images):
	Color transforms, gradients or other methods to create thresholded binary images.
	"""

	# Filter HLS layers, compute x gradient, create a thresholded binary image
	# Add binary to the binary images list
	# Save a copy of the binary image to "output_images\\test_images_thresh_binary" for reference
	binary = functions.threshold_image(destination, combo_thresh=(0, 0), color_thresh=(170, 255), sobelx_thresh=(20, 100))

	# # Uncomment for test image processing
	# plt.imsave('output_images\\test_images_thresh_binary\\' + image_name, binary, cmap='gray')


	"""
	Pipeline (test images):
	Perspective transform to "birds eye" view.
	"""

	# Warp the binary image generated to a "birds eye" view and add it to the list of warped images
	# Save a copy of the warped binary image to "output_images\\test_images_bird_eye" for reference
	warped, Minv = functions.warp_image(binary)

	# # Uncomment for test image processing
	# plt.imsave('output_images\\test_images_bird_eye\\' + image_name, warped, cmap='gray')


	"""
	Pipeline (test iamges):
	Identify lane line pixels, fit a polynomal to the lane lines.
	"""

	# Use a sliding window search to locate the lane lines in the image
	lane_lines_image, ploty, left_fitx, right_fitx = functions.find_lane_lines(warped)

	# # Uncomment for test image processing
	# plt.imsave('output_images\\test_images_detected_lines\\' + image_name, lane_lines_image, cmap='gray')


	"""
	Pipeline (work in progress...):
	Sanity check of data, drop any bad data and average last n number of frames.
	"""

	# Comment for test image processing
	lines.sanity_check(left_fitx, right_fitx)


	"""
	Pipeline (test images):
	Compute a radius to the curvature of the polynomial fit.
	Find the center of the car in between the lanes.
	"""

	# Compute the curvature of the road
left_curve, center_offset, right_curve = functions.measure_curve(lane_lines_image,
								ploty,
								lines.latest_left_fitx,
								lines.latest_right_fitx)

	# # Uncomment for test image processing
	# curves_and_centers.append([left_curve, center_offset, right_curve])


	"""
	Pipeline (test images):
	Plot the detected lane lines back down on to the road image to show the final result.
	"""

	# Plot the final lane detection result back on to the original image
	result = functions.plot_lane(destination,
					warped,
					ploty,
					lines.latest_left_fitx,
					lines.latest_right_fitx,
					Minv,
					left_curve,
					right_curve,
					center_offset)

	# # Uncomment for test image processing
	# plt.imsave('output_images\\test_images_final\\' + image_name, result)

	# Return the plotted image
	return result


# # Uncomment for test image processing
# Read in all the test images, set up a list for undistorted images, binary images, etc
# images = glob.glob('test_images\\*.jpg')
# for f in images:
# 	image_name = f.split('\\')[-1]
# 	img = mpimg.imread(f)
# 	process_image(img)

# Instantiate the line class
lines = Line()

# Generate the completed project video
output = 'project_video_completed.mp4'
clip = VideoFileClip('project_video.mp4')
output_clip = clip.fl_image(process_image)
output_clip.write_videofile(output, audio=False)
