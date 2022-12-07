import cv2
import numpy as np

from imutils.object_detection import non_max_suppression
from tqdm import tqdm


class ObjectRemover:
    def remove_largest_object(self, frames):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        new_frames = np.copy(frames)
        for fn in tqdm(range(frames.shape[0])):
            rects, weights = hog.detectMultiScale(frames[fn],
                                                         winStride=(8, 8),
                                                         padding=(2, 2),
                                                         scale=1.05)
            if not np.any(rects):
                continue

            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            # boxes = non_max_suppression(boxes, probs=None, overlapThresh=0.65)
            areas = np.array([(coord[2] - coord[0]) * (coord[3] - coord[1]) for coord in boxes])
            indices = np.argsort(areas)
            boxes = boxes[indices]

            x1, y1, x2, y2 = boxes[-1]
            new_frames[fn, y1:y2, x1:x2, :] = 0

        return new_frames