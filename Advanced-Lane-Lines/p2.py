import os
import cv2
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pylab as pylab
from scipy.misc import imresize

pylab.rcParams['figure.figsize'] = (14, 6)
font = cv2.FONT_HERSHEY_COMPLEX

def overlay_image(img, s_img, pos, size=None):
    x_offset,y_offset=pos
    l_img = img.copy()
    if size is not None:
        s_img = imresize(s_img, size)
    if len(s_img.shape) == 2:
        s_img = cv2.merge((s_img,s_img,s_img))
    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
    return l_img

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_images(images, row=1, im_prep=None, titles=None, cmap=None):
    add_col = 1 if (len(images) % row) else 0
    col = (len(images) // row) + add_col
    fig, axes = subplots(row, col, subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.05)
    if im_prep is None:
        im_prep = lambda x: x

    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        img = im_prep(images[i])
        if (len(img.shape) < 3) and cmap is None:
            cmap = 'gray'
        ax.imshow(img, cmap=cmap)
        if titles is not None:
            ax.set_title(titles[i])
    show()


class BirdEyeTransform():
    def __init__(self):
        self.dist = np.array([[-0.24688507, -0.02373154, -0.00109831, 0.00035107, -0.00259869]])
        self.mtx = np.array([
            [  1.15777818e+03,   0.00000000e+00,   6.67113857e+02],
            [  0.00000000e+00,   1.15282217e+03,   3.86124583e+02],
            [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
        self.camera_M = np.array([
            [-4.09514513e-01, -1.54251784e+00, 9.09522880e+02],
            [-3.55271368e-15, -1.95774942e+00, 8.94691487e+02],
            [-3.57786717e-18, -2.38211439e-03, 1.00000000e+00]])

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def warp_camera(self, img, inv=False):
        if len(img.shape) == 3:
            img_size = img.shape[-2::-1]
        else:
            img_size = img.shape[::-1]
        flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
        if inv:
            flags += cv2.WARP_INVERSE_MAP
        return cv2.warpPerspective(img, self.camera_M, img_size, flags=flags)

    def transform(self, img, inv=False):
        """
        returns undistorted and warped img
        """
        undist = self.undistort(img)

        warp = self.warp_camera(undist, inv)
        return warp


class ImageSobelBinarizer():
    def binarize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradx = self.abs_sobel_threshold(gray, 'x', 5, thresh=(30, 100))
        grady = self.abs_sobel_threshold(gray, 'y', 15, thresh=(20, 100))
        mg_thresh = self.mag_thresheld(gray, sobel_kernel=5, mag_thresh=(15, 70))
        dir_thresh = self.dir_threshold(gray, sobel_kernel=15, thresh=(-0.1, .8))

        combined = np.zeros(img.shape[:2])
        combined[((gradx == 1) & (grady != 1)) | ((mg_thresh == 1) & (dir_thresh == 1))] = 1
        return combined

    def abs_sobel_threshold(self, gray, orient='x', sobel_kernel=3, thresh=(20, 100)):
        # sobel threshold
        dx = 1 if orient == 'x' else 0
        dy = 1 if orient != 'x' else 0
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        thresh_min = thresh[0]
        thresh_max = thresh[1]
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return sxbinary

    def mag_thresheld(self, gray, sobel_kernel=3, mag_thresh=(20, 100)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output


class ImageHlsBinarizer():
    def __init__(self):
        pass

    def binarize(self, img):
        return self.ls_threshold(img)

    def ls_threshold(self, img, l_thresh_min=40, s_thresh_min=100):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l = hls[:, :, 1]
        s = hls[:, :, 2]
        ret = np.zeros_like(s, dtype=np.ubyte)
        ret[(l > l_thresh_min) & (s > s_thresh_min)] = 1
        return ret


def region_of_interest(img):
    """
    Applies an image mask on warped image

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    h, w = img.shape[:2]
    reg_bl = [220, h]
    reg_br = [w - 220, h]
    reg_tl = [100, 0]
    reg_tr = [w - 100, 0]
    vertices = np.array([[reg_bl, reg_tl, reg_tr, reg_br]], dtype=np.int32)
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

class ImageSobelHlsBinarizer():
    def __init__(self):
        self.birdEyeTransform = BirdEyeTransform()
        self.sobelBinarizer = ImageSobelBinarizer()
        self.hlsBinarizer = ImageHlsBinarizer()

    def binarize(self, frame):
        img = self.birdEyeTransform.transform(frame)
        sobel = self.sobelBinarizer.binarize(img)
        hls_bin = self.hlsBinarizer.binarize(img)
        binary = np.zeros(img.shape[:2])
        binary[(hls_bin == 1) | (sobel == 1)] = 255
        binary = region_of_interest(binary)
        return binary


    def transform(self, frame):
        return self.birdEyeTransform.transform(frame)

    def undistort(self, frame):
        return self.birdEyeTransform.undistort(frame)

    def unwarp(self, img):
        return self.birdEyeTransform.warp_camera(img, True)


class Line():
    def __init__(self, LaneName, margin=100):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        self.curve = None
        self.curven = 0
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        self.margin = margin
        self.base = 0
        self.fitx = None
        self.minpix = 50
        self.nwindows = 9
        self.ploty = None
        self.curve_margin = 1300
        # Define conversions in x and y from pixels space to meters
        self.lane_name = LaneName
        self.n = 15


    def _process_window(self, win_y_low, win_y_high, nonzero):
        # Identify window boundaries in x and y (and right and left)
        win_x_low = self.base - self.margin
        win_x_high = self.base + self.margin
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
            nonzerox < win_x_high)).nonzero()[0]
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > self.minpix:
            self.base = np.int(np.mean(nonzerox[good_inds]))
        return good_inds

    def _slide_window(self, frame_bin):
        nonzero = frame_bin.nonzero()
        window_height = frame_bin.shape[0] // self.nwindows
        lane_inds = []
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = frame_bin.shape[0] - (window + 1) * window_height
            win_y_high = frame_bin.shape[0] - window * window_height
            inds = self._process_window(win_y_low, win_y_high, nonzero)
            lane_inds.append(inds)
        lane_inds = np.concatenate(lane_inds)
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        self.fit = np.polyfit(y, x, 2)
        self.curve = self._measure_curvative(self.fit, x, y)
        self.allx = x
        self.ally = y
        return (self.fit, True)

    def _find_new(self, frame_bin):
        fit = self._slide_window(frame_bin)
        return fit

    def _find(self, frame_bin):
        nonzero = frame_bin.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]
        lane_inds = (
            (nonzerox > (self.fit[0] * (nonzeroy ** 2) + self.fit[1] * nonzeroy + self.fit[2] - self.margin)) & (
            nonzerox < (self.fit[0] * (nonzeroy ** 2) + self.fit[1] * nonzeroy + self.fit[2] + self.margin)))
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        fit = np.polyfit(y, x, 2)
        is_curv = self._check_and_measure_curvative(fit, x, y)
        if is_curv:
            self.allx = x
            self.ally = y
            return (fit, True)
        return (self.fit, False)

    def _measure_curvative(self, fit, x=None, y=None):
        y_eval = np.max(self.ploty)
        curverad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

        # got the coefficient by checking the value curverad_meters/curverad
        curverad_meters = curverad * 0.327
        # Fit new polynomials to x,y in world space
        # self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        # self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # fit_cr = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
        # # Calculate the new radii of curvature
        # curverad_meters = ((1 + (2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        return (curverad, curverad_meters)

    def _check_and_measure_curvative(self, fit, x, y):
        curve = self._measure_curvative(fit, x, y)
        if abs(curve[0] - self.curve[0]) < self.curve_margin:
            self.curve = curve
            return True
        return False

    def append_data(self, fit):
        self.current_fit.append(fit)
        self.current_fit = self.current_fit[-self.n:]
        self.best_fit = np.average(self.current_fit, axis=0)
        fit = self.best_fit
        self.best_fitx = fit[0] * self.ploty ** 2 + fit[1] * self.ploty + fit[2]
        self.curven = self._measure_curvative(fit)

    def update(self, frame_bin, base=None, ploty=None, found=True):
        if ploty is not None:
            self.ploty = ploty
        if found:
            if base is not None:
                self.base = base
            if self.detected:
                fit, found = self._find(frame_bin)
            else:
                fit, found = self._find_new(frame_bin)
            self.fitx = fit[0] * self.ploty ** 2 + fit[1] * self.ploty + fit[2]
            if found:
                self.append_data(fit)
        self.detected = found


class LinesSearch():
    def __init__(self, image_binarizer, windows_count=10):
        self.left_line = Line('Left')
        self.right_line = Line('Right')
        self.windows_count = windows_count
        self.frame = None
        self.binary = None
        self.image_binarizer = image_binarizer
        self.ploty = None

    def _is_line_detected(self):
        """
        :return: True if both lines were detected on previous run
        """
        return self.left_line.detected and self.right_line.detected

    def _get_ploty(self):
        if self.ploty is None:
            self.ploty = np.linspace(0, self.frame.shape[0], self.frame.shape[0], endpoint=False)
        return self.ploty

    def _detect_lines_base(self):
        """
        looking for the line bases on a new image
        :return: (left base, right base)
        """
        half_height = self.binary.shape[0] // 2
        half_frame = self.binary[half_height:, :]
        self.histogram = np.sum(half_frame, axis=0)
        midpoint = self.histogram.shape[0] // 2
        left_base = np.argmax(self.histogram[:midpoint])
        right_base = np.argmax(self.histogram[midpoint:]) + midpoint
        return (left_base, right_base)

    def _detect_lines(self):
        """
        depending on the current state, the method is looking for the lines
        :return:
        """
        left_base = None
        right_base = None
        if not self._is_line_detected():
            left_base, right_base = self._detect_lines_base()
        self.left_line.update(self.binary, left_base, self._get_ploty())
        self.right_line.update(self.binary, right_base, self._get_ploty())
        if not self._check_curvative():
            return False
        return True

    def _check_curvative(self):
        """
        if radius's diffrence is more than 300m returns False
        :return:
        """
        l_curvative = self.left_line.curve[1]
        r_curvative = self.right_line.curve[1]
        if (abs(l_curvative - r_curvative) > 300):
            return False
        return True

    def plot(self, show=False):
        """
        Drawing the lane on the initial frame
        :param show:
        :return:
        """
        left_fitx = self.left_line.best_fitx
        right_fitx = self.right_line.best_fitx
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        color_warp = np.zeros(self.frame.shape, dtype=np.ubyte)
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        undist_frame = self.image_binarizer.undistort(self.frame)
        unwarp = self.image_binarizer.unwarp(color_warp)
        ret = cv2.addWeighted(undist_frame, 1, unwarp, 0.3, 0)
        ret = overlay_image(ret,  self.image_binarizer.transform(self.frame), (0, 0), (200, 400))
        ret = overlay_image(ret, self.binary, (402, 0), (200, 400))

        curvr = curvl = "---m"
        if self.left_line.curven[1] < 2500:
            curvl = "{:.2f}m".format(self.left_line.curven[1])
        if self.right_line.curven[1] < 2500:
            curvr = "{:.2f}m".format(self.right_line.curven[1])
        textSize,_ = cv2.getTextSize(curvr, font, 1, 2)
        cv2.putText(ret, curvl, (0, ret.shape[0]-textSize[1]), font, 1, (255, 0, 0), 2)
        cv2.putText(ret, curvr, (ret.shape[1]-textSize[0], ret.shape[0]-textSize[1]), font, 1, (0, 0, 255), 2)

        left_bx = np.average(left_fitx[-10:])
        right_bx = np.average(right_fitx[-10:])
        lane_width = 0.00755 * abs(right_bx - left_bx)
        cv2.putText(ret, "Lane width:{:.2f}".format(lane_width), (0, ret.shape[0]-3*textSize[1]), font, 1, (255, 0, 255), 2)

        lane_mid = (right_bx - left_bx) / 2 + left_bx
        middle = self.binary.shape[1] / 2
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        offset = abs(lane_mid-middle)*xm_per_pix
        offset_str = "offset: {:.2f}m".format(offset)
        textSize,_ = cv2.getTextSize(offset_str, font, 1, 2)
        cv2.putText(ret, offset_str, (ret.shape[1]-textSize[0], ret.shape[0]-3*textSize[1]), font, 1, (255, 0, 255), 2)
        if show:
            plot_images(((self.frame), (ret)), im_prep=bgr2rgb)
        return ret

    def _binarize(self, frame):
        return self.image_binarizer.binarize(frame)

    def search(self, frame):
        self.frame = frame
        self.binary = self._binarize(frame)
        self._detect_lines()

if __name__ == '__main__':
    ls = LinesSearch(ImageSobelHlsBinarizer())

    vid = cv2.VideoCapture('project_video.mp4')
    while vid.isOpened():
        ret, camera_img = vid.read()
        if ret is None:
            break
        # camera_img = cv2.imread('test_images/test5.jpg')
        # camera_img = cv2.imread('test_images/straight_lines1.jpg')
        ls.search(camera_img)
        # ls.plot(True)
        # continue
        result = ls.plot()
        cv2.imshow('all_result', result)
        stop = False
        # while True:
        if 1:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                break
            if key == ord(' '):
                stop = True
                break
        if stop:
            break

    vid.release()
    cv2.destroyAllWindows()


