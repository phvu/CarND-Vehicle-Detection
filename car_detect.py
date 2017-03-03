import argparse
import multiprocessing
import os

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label as scipy_label
from sklearn.externals import joblib

from car_detect_train import get_hog_features, bin_spatial, color_hist


def find_cars(img, ystart, ystop, scale, ppl, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    candidates = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_prediction = ppl.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                candidates.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                cv2.rectangle(draw_img, candidates[-1][0], candidates[-1][1], (0, 0, 255), 6)

    return candidates, draw_img


def get_labeled_bboxes(labels):
    """
    Use the labels arrays to extract bounding boxes of detected cars
    Return a list of bounding boxes ((x1, y1), (x2, y2))
    """
    bboxes = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)

    return bboxes


def filter_candidates(img, candidates, cars, heat_threshold=1, max_smooth=10, min_overlap=0.6, min_area=1000):
    """
    Given the list of candidates detected by sliding windows,
    compute the heat map and extract the "canonical" car positions in the image
    `cars` is a list of CarPosition objects
    """
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # add heat
    for box in candidates:
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Apply threshold to help remove false positives
    heat[heat <= heat_threshold] = 0

    # Visualize the heatmap
    heat = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = scipy_label(heat)
    bboxes = get_labeled_bboxes(labels)

    # assign the bboxes to the cars
    for bbox in bboxes:
        if bbox_area(bbox) < min_area:
            continue

        if len(cars) == 0:
            cars = [CarPosition(bbox, max_smooth=max_smooth)]
        else:
            overlaps = [c.overlap(bbox) for c in cars]
            idx = np.argmax(overlaps)
            if overlaps[idx] >= min_overlap:
                cars[idx].update(bbox)
            else:
                cars.append(CarPosition(bbox, max_smooth=max_smooth))

    # Draw the box on the image
    car_img = np.copy(img)
    for car in cars:
        pos = car.position
        cv2.rectangle(car_img, pos[0], pos[1], (0, 0, 255), 6)

    return cars, car_img, heat


def bbox_area(bbox=((0, 0), (2, 6))):
    """
    Compute area of bbox
    """
    dx = bbox[1][0] - bbox[0][0]
    dy = bbox[1][1] - bbox[0][1]
    if dx < 0 or dy < 0:
        raise ValueError('Invalid bbox: {}'.format(bbox))
    return dx * dy


class CarPosition(object):
    def __init__(self, bbox, max_smooth=10):
        self.bboxes = np.array([np.array(bbox).ravel()])
        self.max_smooth = max_smooth

    @property
    def position(self):
        """
        Return the adjusted position of this car in the format ((x1, y1), (x2, y2))
        """
        pos = [int(_) for _ in self.bboxes.mean(0)]
        return (pos[0], pos[1]), (pos[2], pos[3])

    def overlap(self, bbox):
        """
        Compute the percentage of overlapping between the given bbox and the adjusted position of this car
        Return a float in [0, 1]
        """
        pos = self.position
        dx = min(pos[1][0], bbox[1][0]) - max(pos[0][0], bbox[0][0])
        dy = min(pos[1][1], bbox[1][1]) - max(pos[0][1], bbox[0][1])
        if dx >= 0 and dy >= 0:
            intersect = dx * dy
            return intersect / (min(bbox_area(pos), bbox_area(bbox)))
        return 0

    def update(self, bbox):
        """
        Update the car position with the new bbox
        """
        new_bbox = np.array(bbox).ravel()
        self.bboxes = np.vstack((self.bboxes, new_bbox))
        if len(self.bboxes) > self.max_smooth:
            self.bboxes = self.bboxes[-self.max_smooth:, :]


class CarDetectionModel(object):
    def __init__(self, model_file='car_model.pkl'):
        persisted_data = joblib.load(model_file)
        self.ppl = persisted_data['ppl']
        self.orient = persisted_data['orient']
        self.pix_per_cell = persisted_data['pix_per_cell']
        self.cell_per_block = persisted_data['cell_per_block']
        self.spatial_size = persisted_data['spatial_size']
        self.hist_bins = persisted_data['hist_bins']


class CarFeatureDetector(multiprocessing.Process):
    def __init__(self, work_queue, output_queue):
        self.work_queue = work_queue
        self.output_queue = output_queue
        super(CarFeatureDetector, self).__init__()

    def run(self):
        detection_model = CarDetectionModel()
        for request in iter(self.work_queue.get, None):
            img, ystart, ystop, scale = request
            c, _ = find_cars(img, ystart, ystop, scale, detection_model.ppl,
                             detection_model.orient, detection_model.pix_per_cell, detection_model.cell_per_block,
                             detection_model.spatial_size, detection_model.hist_bins)
            self.output_queue.put(c)


class CarDetector(object):
    def __init__(self, debug=False, workers=4):
        self.debug = debug
        self.detection_model = CarDetectionModel()

        self.ystart = 400
        self.ystop = 656
        self.scales = (0.8, 1, 1.25, 1.5, 1.7, 2)
        self.heat_threshold = 5
        self.cars = []
        self.max_smooth = 10
        self.min_overlap = 0.3
        self.min_area = 1000

        self.work_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.detectors = [CarFeatureDetector(self.work_queue, self.output_queue) for _ in range(workers)]
        [d.start() for d in self.detectors]

    def get_candidates_serial(self, img):
        candidates = []
        for scale in self.scales:
            c, _ = find_cars(img, self.ystart, self.ystop, scale, self.detection_model.ppl,
                             self.detection_model.orient, self.detection_model.pix_per_cell,
                             self.detection_model.cell_per_block,
                             self.detection_model.spatial_size, self.detection_model.hist_bins)
            candidates.extend(c)
        return candidates

    def get_candidates_parallel(self, img):
        for scale in self.scales:
            self.work_queue.put((img, self.ystart, self.ystop, scale))

        candidates = []
        for _ in self.scales:
            c = self.output_queue.get()
            candidates.extend(c)
        return candidates

    def detect(self, img):
        assert (len(img.shape) == 3 and img.shape[2] == 3 and img.dtype == np.uint8)

        candidates = self.get_candidates_parallel(img)
        self.cars, car_img, heat_img = filter_candidates(img, candidates, self.cars,
                                                         self.heat_threshold, self.max_smooth, self.min_overlap,
                                                         self.min_area)
        return car_img, heat_img, self.cars

    def detect_video(self, inp_file, out_file):

        def detect_func(frame):
            car_img, heat_img, _ = self.detect(frame)
            if self.debug:
                heat_img = (heat_img - heat_img.min()) / (heat_img.max() - heat_img.min())
                heat_img = (heat_img * 255).astype(np.uint8)
                return np.hstack((car_img, np.dstack((heat_img, np.zeros_like(heat_img), np.zeros_like(heat_img)))))
            return car_img

        out_clip = VideoFileClip(inp_file).fl_image(detect_func)
        out_clip.write_videofile(out_file, audio=False, verbose=False)

        self.work_queue.put(None)
        for p in self.detectors:
            p.terminate()
            p.join()

        return out_clip


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='Detecting cars')
    parser.add_argument('input', type=str, help='Path to the input video')
    parser.add_argument('--out', '-o', dest='output', type=str, default='', help='Path to the output video')
    parser.add_argument('--debug', '-d', dest='debug', action='store_true')
    parser.add_argument('--process', '-p', dest='process', type=int, default=4, help='Number of worker processes')

    args = parser.parse_args()

    detector = CarDetector(debug=args.debug, workers=args.process)
    if args.output == '':
        args.output = '{}_detected{}'.format(*os.path.splitext(args.input))
    detector.detect_video(args.input, args.output)
