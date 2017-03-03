import argparse
import glob
import os
import pickle

import cv2
import numpy as np
from moviepy.editor import VideoFileClip


class Lane(object):
    def __init__(self, left_fit, right_fit, left_fitx, right_fitx, leftx, rightx, lefty, righty):
        self.left_fit = left_fit
        self.right_fit = right_fit
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

    def _curvature(self):
        """
        Curvature in the pixel world
        :return:
        """
        y_eval = self.left_fitx.shape[0] - 10
        left_curverad = (((1 + (2 * self.left_fit[0] * y_eval + self.left_fit[1]) ** 2) ** 1.5) /
                         np.absolute(2 * self.left_fit[0]))
        right_curverad = (((1 + (2 * self.right_fit[0] * y_eval + self.right_fit[1]) ** 2) ** 1.5) /
                          np.absolute(2 * self.right_fit[0]))
        return left_curverad, right_curverad

    def sanity_check(self):
        """
        Check if this lane is good enough...
        :return: score
        """
        score = 0
        curvatures = self._curvature()
        if abs(curvatures[0] - curvatures[1]) / max(curvatures) > 0.15:
            # difference in curvature is more than 15%
            score -= 1

        diff_std = np.std(self.right_fitx - self.left_fitx)
        if diff_std > 30:
            # std of the difference between the right lane and left lane is more than 30 pixel
            score -= 1

        # roughly parallel
        if abs(self.left_fit[0] - self.right_fit[0]) / max(self.left_fit[0], self.right_fit[0]) > 0.15:
            # difference in slope is more than 15%
            score -= 1

        return score


def calibrate():
    """
    Calibrate camera using a set of chessboard images
    """
    if os.path.exists('calibration_data.pkl'):
        with open('calibration_data.pkl', 'rb') as f:
            return pickle.load(f)

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in glob.glob('camera_cal/calibration*.jpg'):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print('{}: {}'.format(fname, gray.shape))

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print('Failed to detect corners for {}'.format(fname))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
    assert ret

    with open('calibration_data.pkl', 'wb') as f:
        pickle.dump((mtx, dist), f)

    return mtx, dist


def undistort(img, mtx, dist):
    """
    Undistort the input image, using the given mtx and dist matrices
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def binarize(img, s_thres=(170, 255), l_thres=(50, 255), sobel_thres=(30, 80)):
    """
    Input image should be in BGR
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hls[:, :, 1] = clahe.apply(hls[:, :, 1])

    l_image = hls[:, :, 1]
    l_blur = cv2.GaussianBlur(l_image, (0, 0), 9)
    l_image = cv2.addWeighted(l_image, 1, l_blur, -1, 0)
    l_image = cv2.normalize(l_image, np.zeros_like(l_image), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    l_binary = np.zeros_like(l_image)
    l_binary[(l_image >= l_thres[0]) & (l_image <= l_thres[1])] = 1

    # Sobel x
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = hls[:, :, 1]
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    # scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # sxbinary = np.zeros_like(scaled_sobel)
    # sxbinary[(scaled_sobel >= sobel_thres[0]) & (scaled_sobel <= sobel_thres[1])] = 1
    # sxbinary = s_binary

    s_channel = hls[:, :, 2]
    s_channel = cv2.normalize(s_channel, np.zeros_like(s_channel), 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thres[0]) & (s_channel <= s_thres[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (l_binary == 1)] = 1

    # we filter out the lines with too many active pixels
    combined_binary_rows = combined_binary.sum(1)
    combined_binary[combined_binary_rows > (combined_binary.shape[1] / 2)] = 0

    return combined_binary


def get_transform():
    src = np.float32([
        [595, 450],
        [685, 450],
        [1040, 674],
        [269, 674]])
    dst = np.float32([
        [200, 10],
        [1000, 10],
        [1000, 700],
        [200, 700]])
    return cv2.getPerspectiveTransform(src, dst)


def warp(transform_mat, img):
    # keep same size as input image
    return cv2.warpPerspective(img, transform_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)


def detect_lanes(binary_warped):
    """
    binary_warped is a binary image
    """
    histogram = np.sum(binary_warped[(binary_warped.shape[0] // 2):, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9

    window_height = np.int(binary_warped.shape[0] / nwindows)

    nonzero = binary_warped.nonzero()
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
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

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

    # Visualize
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return Lane(left_fit, right_fit, left_fitx, right_fitx, leftx, rightx, lefty, righty), out_img


def infer_lanes(binary_warped, left_fit, right_fit):
    """
    binary_warped: binary image
    """
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 50
    left_line = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2]
    right_line = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2]
    left_lane_inds = ((nonzerox > (left_line - ((2 if left_fit[0] > 0 else 1) * margin))) &
                      (nonzerox < (left_line + ((2 if left_fit[0] < 0 else 1) * margin))))
    right_lane_inds = ((nonzerox > (right_line - ((2 if right_fit[0] > 0 else 1) * margin))) &
                       (nonzerox < (right_line + ((2 if right_fit[0] < 0 else 1) * margin))))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Visualize
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

    return Lane(left_fit, right_fit, left_fitx, right_fitx, leftx, rightx, lefty, righty)


def compute_curvature(left_fitx, right_fitx, xm_per_pix, ym_per_pix):
    """
    Compute curvature in the real world
    :return: curvatures in meters
    """
    # Fit new polynomials to x,y in world space
    ploty = np.linspace(0, left_fitx.shape[0] - 1, left_fitx.shape[0])
    y_eval = left_fitx.shape[0] - 10

    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = (((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) /
                     np.absolute(2 * left_fit_cr[0]))
    right_curverad = (((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) /
                      np.absolute(2 * right_fit_cr[0]))
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


def compute_position(left_fitx, right_fitx, inverse_transform, xm_per_pix, img_width):
    unwarped_loc = np.dot(inverse_transform, [[left_fitx[-1], right_fitx[-1]],
                                              [left_fitx.shape[0] - 1, right_fitx.shape[0] - 1],
                                              [1, 1]])
    assert (unwarped_loc.shape == (3, 2))
    unwarped_loc[:-1, :] /= unwarped_loc[-1, :]
    return ((img_width / 2) - ((unwarped_loc[0, 1] + unwarped_loc[0, 0]) / 2)) * xm_per_pix


def draw_lane(left_fitx, right_fitx, binary_warped):
    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    lane = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), np.uint8)
    cv2.fillPoly(lane, np.int_([pts]), (255, 255, 0))

    return lane


class LaneDetector(object):
    def __init__(self, debug=False):
        self.mtx, self.dist = calibrate()
        self.transform_matrix = get_transform()
        self.inverse_transform = np.zeros(self.transform_matrix.shape[::-1], self.transform_matrix.dtype)
        assert (cv2.invert(self.transform_matrix, self.inverse_transform) != 0)
        self.lane = None
        self.debug = debug

        self.xm_per_pix = 3.7 / 800  # meters per pixel in x dimension
        self.ym_per_pix = 40 / 690  # meters per pixel in y dimension

    def preprocess(self, img):
        """
        img should be a BGR image
        Returns the binary warped image
        """
        undistorted = undistort(img, self.mtx, self.dist)
        warped = warp(self.transform_matrix, undistorted)
        binary_img = binarize(warped, s_thres=(100, 255), l_thres=(50, 255))
        return binary_img

    def process_image(self, image, draw_img):
        warped_binary = self.preprocess(image[:, :, ::-1])

        if self.lane is None:
            self.lane, _ = detect_lanes(warped_binary)
        else:
            self.lane = infer_lanes(warped_binary, self.lane.left_fit, self.lane.right_fit)

        curvatures = compute_curvature(self.lane.left_fitx, self.lane.right_fitx,
                                       self.xm_per_pix, self.ym_per_pix)

        lanes_img = draw_lane(self.lane.left_fitx, self.lane.right_fitx, np.zeros_like(warped_binary))
        unwarped_lanes = warp(self.inverse_transform, lanes_img)
        result = cv2.addWeighted(draw_img, 1, unwarped_lanes, 0.4, 0)

        # write the curvatures
        cv2.rectangle(result, (0, 0), (result.shape[1], 100), (100, 100, 100), -1)
        cv2.putText(result, 'Estimated curvatures: {:.2f}m {:.2f}m'.format(*curvatures),
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # vehicle position
        pos = compute_position(self.lane.left_fitx, self.lane.right_fitx, self.inverse_transform,
                               self.xm_per_pix, draw_img.shape[1])
        cv2.putText(result, 'Vehicle relative position to the middle of the lane: {:.2f}m'.format(pos),
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(result, 'Score: {}'.format(self.lane.sanity_check()),
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if self.debug:
            b = np.dstack((warped_binary * 255, warped_binary * 255, warped_binary * 255))
            result = np.hstack((result, cv2.addWeighted(b, 1, lanes_img, 0.6, 0)))
        return result
