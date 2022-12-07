import cv2
import numpy as np


class DrawLineWidget:

    def __init__(self,
                 image: np.ndarray,
                 video: np.ndarray,
                 name: str = 'camera'):
        self.original_image = image
        self.clone = self.original_image.copy()
        self.window_name = name
        self._video = video

        cv2.namedWindow(name)
        cv2.setMouseCallback(name, self.extract_coordinates)

        # List to store start/end points
        self.image_coordinates = [(0, 0), (0, 0)]
        self._tmp = []

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clone = self.original_image.copy()
            self._tmp = [(x, y)]

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self._tmp.append((x, y))
            self.image_coordinates = self._tmp[:2]
            print(self.image_coordinates)
            # Draw line
            cv2.line(self.clone, self.image_coordinates[0],
                     self.image_coordinates[1], (36, 255, 12), 2)

        cv2.imshow(self.window_name, self.clone)

    def show_image(self):
        return self.clone
