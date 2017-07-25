# Udacity CARND P4 helper functions
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def calibrate_camera(cal_images='camera_cal\\', image_type='.jpg', nx=9, ny=6):
	"""
	Computes the camera calibration matrix and distortion coefficients.

	cal_images: path to images to be used for calibration (string)
	image_type: type of image IE: '.jpg' (string)
	nx: number of cal_images inside corners for x (int)
	ny: number of cal_images inside corners for y (int)
	"""

	# Make a list of calibration images
	images = glob.glob(cal_images + '*' + image_type) # get the names of all the images in the folder of the specified type

	# Columns for x,y,z values
	cols = 3

	objpoints = [] # 3D points in real world space
	imgpoints = [] # 2D points in image space

	# Initialize the object points for the chessboard in an array
	objp = np.zeros(((nx) * (ny), cols), np.float32)

	# Create x,y coordinates for chessboard array, (np.mgrid reshape, z value will always be 0 when using only one image plane)
	objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

	# Convert all the images to grayscale, detect the corners, calibrate the camera, and undistort the images
	for f in images:
		# Read each image, convert to grayscale in RGB format
		img = mpimg.imread(f)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# Detect the chessboard corners on the grayscale image
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		if ret == True:
			# If findChessboardCorners returns True, add the "objp" to the object points list
			# and add the corresponding image corners detected to the image points list
			objpoints.append(objp)
			imgpoints.append(corners)

			# Draw the corners, save in "output_images/camera_cal_image_corner_detection" for reference
			img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
			plt.imsave('output_images\\camera_cal_image_corner_detection\\' + f.split('\\')[-1], img)

			# Calibrate the camera with the detected corner points
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

			# Undistort the image, save in "output_images/camera_cal_undistorted" for reference
			destination = cv2.undistort(img, mtx, dist, None, mtx)
			plt.imsave('output_images\\camera_cal_undistorted\\' + f.split('\\')[-1], destination)

	# Return a distortion calibration
	return mtx, dist


def threshold_image(img, combo_thresh=(0, 255), color_thresh=(0, 255), sobelx_thresh=(0, 255)):
	"""
	Selects HLS color spaces to create a binary image that extracts lane lines.
	Applies a Sobel X gradient to the image to create a binary image that extracts lane lines.

	Overlays the two binary images to create the final result.

	img: image to extract lane lines from (numpy array)
	color_thresh: thresholds for color channels (tuple of ints (min,max))
	sobelx_thresh: thresholds for sobel gradient in the x direction (tuple of ints (min,max))

	combo_thresh: thresholds the REMAINING pixel values from the layered
	sobelx and color binaries (tuple of ints (sobelx pixel value, color pixel value))
	"""

	# Copy the image
	img = np.copy(img)

	# Convert to HLS color space and separate the L and S channels
	HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
	L = HLS[:,:,1]
	S = HLS[:,:,2]

	# Take the derivative in the x (horizontal) direction on just the L channel of HSL
	sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0) # take the derivative in x
	abs_sobelx = np.absolute(sobelx) # take the ABS value of sobelx to get the result
	scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx)) # convert to 8 bit image for convenience later

	# Threshold x gradient from Sobel(x) of the L channel of HLS
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sobelx_thresh[0]) & (scaled_sobel <= sobelx_thresh[1])] = 1

	# Threshold color channel on just S of HSL
	c_binary = np.zeros_like(S)
	c_binary[(S >= color_thresh[0]) & (S <= color_thresh[1])] = 1

	# Overlay the channels to create a complete binary image
	color_binary = np.dstack((sxbinary, c_binary))

	# Convert the color layered binary to black and white and threshold remaining pixel value combos
	# for sobel and color binaries
	binary = np.zeros_like(sxbinary)
	binary[(color_binary[:,:, 0] > combo_thresh[0]) | (color_binary[:,:, 1] > combo_thresh[1])] = 1

	# Return the final binary
	return binary


def warp_image(img):
	"""
	Warps a binary image to transform it to a birds eye view.

	img: binary mask image (numpy array)
	"""

	# Copy the image
	img = np.copy(img)

	image_size = img.shape[::-1] # get binary image (x,y) shape and reverse to (y,x) for cv2.warpPerspective()

	# Set up 4 points in a trapezoid to transform the lane lines to a "birds eye" view
	tl = [590, 450] # top left
	tr = [690, 450] # top right
	br = [1100, image_size[1]] # bottom right
	bl = [200, image_size[1]] # bottom left

	# Put all of the source coordinates in a numpy array
	src_points = np.float32([tl, tr, br, bl])

	# Set the destination points (abtracted from src_points)
	offset = 250

	dest_points = np.float32([
				[tl[0] - (offset * 1.4), 0],
				[tr[0] + (offset * 1.4), 0],
				[br[0] - (offset // 4), br[1]],
				[bl[0] + (offset // 4), br[1]]])

	# Compute the perspective transform
	M = cv2.getPerspectiveTransform(src_points, dest_points)
	Minv = cv2.getPerspectiveTransform(dest_points, src_points) # reversed the warp


	# Warp the image from src to dst points
	warped = cv2.warpPerspective(img, M, image_size, flags=cv2.INTER_LINEAR)

	# Return the warped image and inverse warp
	return warped, Minv


def find_lane_lines(img):
	"""
	Detects the lane lines in a binary warped image using peaks in "hot" pixel areas
	as well as a sliding window search in a cross section of the x axis.

	img: binary warped image (numpy array)
	"""

	# Copy the image
	img = np.copy(img)

	# Add up the pixels column wise to find peaks in the binary image
	histogram = np.sum(img[img.shape[0]//2:,:], axis=0) # bottom half of image

	# Create an  ouput image to draw on and visualize the result
	out_img = np.dstack((img, img, img))*255

	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = np.int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[:midpoint])
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9

	# Set height of windows
	window_height = np.int(img.shape[0]/nwindows)

	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base

	# Set the width of the windows +/- margin
	margin = 100

	# Set minimum number of pixels found to recenter window
	minpix = 50

	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = img.shape[0] - (window+1)*window_height
		win_y_high = img.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin

		# Draw the windows on the visualization image
		cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
		cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)

		# If you found > minpix pixels, recenter next window on their mean position
		if len(good_left_inds) > minpix:
			leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:
			rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)

	# Generate x and y values for plotting
	ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Change pixel values for detected lane pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Return an image of the detected lane lines
	return out_img, ploty, left_fitx, right_fitx


def measure_curve(lane_lines_image, ploty, left_fitx, right_fitx):
	"""
	Computes the curvature of the road and returns the object space value in meters.

	lane_lines_image: binary warped lane line image with shape (y, x, layers), (numpy array)
	ploty: y value points from an img (all possible y pixel values (0,max y))
	left_fitx: fitted pixel values for left lane lines, indice values correspond to ploty y values (numpy array)
	right_fitx: fitted pixel values for right lane lines, indice values correspond to ploty y values (numpy array)

	IE: (left_fitx[i], ploty[i]) = (x,y) pixel position for lane line center
	"""

	# Define y-value where we want radius of curvature
	# Choose the maximum y-value, corresponding to the bottom of the image
	y_eval = np.max(ploty)

	# Convert pixel space curvature to object space curvature
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension

	# Fit new polynomials to x,y in world space
	left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
	right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

	# Calculate the new radii of curvature in meters
	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

	# Find the center of the car in between the lane lines in a x pixel value, calculate an offset from true center (center of image)
	# Convert to meters
	lane_line_center = (left_fitx[-1] + right_fitx[-1]) / 2 # get x cordinate at bottom of image closest to vehicle
	vehicle_center_offset = round(((lane_lines_image.shape[1] / 2) - lane_line_center) * xm_per_pix, 2)

	# Now that the radius of curvature is in meters, return the curve for the left and right lines
	return left_curverad, vehicle_center_offset, right_curverad


# Line class to keep track of frames and their respective measurements
class Line():
	"""
	A simple line class that is used to keep track of key computed line parameters
	from each frame of a video. In addition, a list of lines are kept for averaging.
	"""

	def __init__(self):

		# Number of past fits to keep for moving averages
		self.frame_count = 10

		# Limit for outlier detected lane line pixel values
		self.outlier_pixel_limit = 25

		# x values of the last n fits of the line
		self.recent_xfitted = []

		# Latest left fit data, could be an adjusted value (see sanity_check())
		self.latest_left_fitx = None

		# Latest right fit data, could be an adjusted value (see sanity_check())
		self.latest_right_fitx = None

	def sanity_check(self, left_fitx, right_fitx):
		# Gather some line data to save if we dont have enough
		if len(self.recent_xfitted) < self.frame_count:
			self.recent_xfitted.append([left_fitx, right_fitx])

			# Update latest right and left line fits
			self.latest_left_fitx = self.recent_xfitted[-1][0]
			self.latest_right_fitx = self.recent_xfitted[-1][1]

		else:
			# Check the data collected before letting it through the pipeline
			# Correct any outliers in data with previous data averages and a set limit
			# Update any attributes to be current

			# Average left and right fits
			# Add each line fit up from the current collected frames in recent_xfitted
			A_left_line_fit = np.mean(np.array([left[0] for left in self.recent_xfitted]), axis=0)
			A_right_line_fit = np.mean(np.array([right[1] for right in self.recent_xfitted]), axis =0)

			#TODO: Implement some frame dropping or weighted averaging to help smooth
			# frames in problematic sections

			# # Compare the averages of the last n fits to the current line
			# # Replaces outlier x points with averaged points when beyound the set limit
			# np.putmask(left_fitx, abs(left_fitx - A_left_line_fit) > self.outlier_pixel_limit, A_left_line_fit)
			# np.putmask(right_fitx, abs(right_fitx - A_right_line_fit) > self.outlier_pixel_limit, A_right_line_fit)

			# Remove the oldest line fit data from the list, add the new
			del self.recent_xfitted[0]
			self.recent_xfitted.append([left_fitx, right_fitx])

			# Update latest right and left line fits
			self.latest_left_fitx =  self.recent_xfitted[-1][0]
			self.latest_right_fitx = self.recent_xfitted[-1][1]


def plot_lane(img, warped, ploty, left_fitx, right_fitx, Minv, left_curve, right_curve, center_offset):
	"""
	Plots the the detected lane back down on to the original undistorted image.

	img: original undistorted image (numpy array)
	warped: warped binary image with the lane lines (numpy array)
	Minv: the cv2.getPerspectiveTransform() with reversed source and destination points
	to unwarp an image

	left_curve: computed left curve in meters (float)
	right_curve: computed right curve in meters (float)
	center_offset: computed vehicle offset from center of frame (float)
	"""

	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

	# Write the center offset to the image
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(newwarp,'Left Curve: {}m'.format(round(left_curve, 2)),(50,50), font, 0.65, (255,255,255), thickness=2)
	cv2.putText(newwarp,'Center Offset: {}m'.format(center_offset),(500,50), font, 0.65, (255,255,255), thickness=2)
	cv2.putText(newwarp,'Right Curve: {}m'.format(round(right_curve, 2)),(950,50), font, 0.65, (255,255,255), thickness=2)

	# Combine the plot with the original image
	result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

	# Return the final modified image with plotted lane lines and center offset
	return result
