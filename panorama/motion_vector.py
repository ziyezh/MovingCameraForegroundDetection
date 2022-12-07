import math
import sys

import cv2
import numpy as np


class MotionVector:

    def getBlockMV(self, prevframe, curframe, y, x, bh, bw, k):
        #our search distance is 2K+1 for k is given by input
        #our right point will be i+16, j+16
        height, width, _ = curframe.shape
        curblock = curframe[y:y + bh, x:x + bw, 0]
        #searching block area
        #left upper point and right down point of searching block area
        xstart, ystart, xend, yend = max(x - k, 0), max(y - k, 0), min(
            x + k, width - bw), min(y + k, height - bh)

        mad_values = []
        for i in range(ystart, yend + 1):
            for j in range(xstart, xend + 1):
                mad_values.append(
                    np.mean(
                        np.absolute(curblock -
                                    prevframe[i:i + bh, j:j + bw, 0])))

        min_index = np.argmin(mad_values)
        target_y, target_x = ystart + min_index // (
            xend - xstart + 1), xstart + (min_index % (xend - xstart + 1))

        return [y - target_y, x - target_x]
