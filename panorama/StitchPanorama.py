import sys

import cv2
import numpy as np
from tqdm import tqdm

from panorama.matcher import matcher


#https://python.plainenglish.io/opencv-image-stitching-second-part-388784ccd1a
#https://towardsdatascience.com/image-panorama-stitching-with-opencv-2402bde6b46c
class StitchPanorama:

    def __init__(self, frames):
        self.images = frames
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = matcher()
        self.prepare_lists()

    def simpleStitch(self):
        stitchy = cv2.Stitcher.create()
        ret, panorama = stitchy.stitch(self.images)
        print("go stitching ...")

        if ret != cv2.STITCHER_OK:
            print('stitch Failed! error: ', ret)
        else:
            return panorama

        # prev = panorama = self.images[0]
        # end = len(self.images)
        # for i in tqdm(range(1, end)):
        #     ret, panorama = stitchy.stitch([ panorama, self.images[i]])
        #     print(cv2.STITCHER_OK, ret)
        #     if ret != cv2.STITCHER_OK:
        #         print('stitch Failed! error: ', ret)
        #         panorama = prev
        #     else:
        #         panorama = cv2.resize(panorama, self.images[0].shape[:2])
        #         prev = panorama

        return prev

    def prepare_lists(self):
        self.centerIdx = self.count / 2
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if i <= self.centerIdx:
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])

    def leftshift(self):
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.matcher_obj.match(a, b, 'left')
            xh = np.linalg.inv(H)
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            ds = ds // ds[-1]
            f1 = np.dot(xh, np.array([0, 0, 1]))
            f1 = f1 // f1[-1]
            xh[0][-1] += abs(f1[0])
            xh[1][-1] += abs(f1[1])
            ds = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            offsety = abs(int(f1[1]))
            offsetx = abs(int(f1[0]))
            print(offsety, offsetx)
            dsize = (int(ds[0]) + offsetx, int(ds[1]) + offsety)
            print(dsize)
            tmp = cv2.warpPerspective(a, xh, dsize)
            print(len(tmp), len(tmp[0]))
            print(b.shape[0], b.shape[1])
            tmp[offsety:b.shape[0] + offsety, offsetx:b.shape[1] + offsetx] = b
            a = tmp

        self.leftImage = tmp

    def rightshift(self):
        for each in self.right_list:
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz // txyz[-1]
            dsize = (int(txyz[0]) + self.leftImage.shape[1],
                     int(txyz[1]) + self.leftImage.shape[0])
            tmp = cv2.warpPerspective(each, H, dsize)
            tmp = self.mix_and_match(self.leftImage, tmp)
            self.leftImage = tmp

    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]
        black_l = np.where(leftImage == np.array([0, 0, 0]))
        black_wi = np.where(warpedImage == np.array([0, 0, 0]))

        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if (np.array_equal(leftImage[j, i], np.array([0, 0, 0]))
                            and np.array_equal(warpedImage[j, i],
                                               np.array([0, 0, 0]))):
                        # instead of just putting it with black,
                        # take average of all nearby values and avg it.
                        warpedImage[j, i] = [0, 0, 0]
                    else:
                        if (np.array_equal(warpedImage[j, i], [0, 0, 0])):
                            warpedImage[j, i] = leftImage[j, i]
                        else:
                            if not np.array_equal(leftImage[j, i], [0, 0, 0]):
                                bl, gl, rl = leftImage[j, i]
                                warpedImage[j, i] = [bl, gl, rl]
                except:
                    pass
        return warpedImage

    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.leftImage)


# cv2.imshow("left image", cv2.resize(self.leftImage, (400,400)))
        elif string == "right":
            cv2.imshow("right Image", self.rightImage)
        cv2.waitKey()

    def getPanorama(self):
        self.leftshift()
        self.rightshift()
        return self.leftImage
