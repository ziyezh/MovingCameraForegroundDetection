import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from tqdm import tqdm

from panorama.motion_vector import MotionVector


class ForegroundExtractor:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def subsample(self, arr):
        row = arr.shape[0]
        col = arr.shape[1]
        depth = arr.shape[2]
        blockSize = 16
        n_row = (row - 1) // blockSize + 1
        n_col = (col - 1) // blockSize + 1
        n = np.zeros((n_row, n_col, depth))
        for i in range(n_row):
            for j in range(n_col):
                for k in range(depth):
                    n[i][j][k] = arr[i * blockSize:(i + 1) * blockSize:, j * blockSize:(j + 1) * blockSize:, k].mean()
        return n

    def get_foreground_mask_grabcut(self, frames):
        fgmasks = []
        for frame in tqdm(frames):
            fgmask = np.zeros(frame.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (0, 0, frame.shape[1], frame.shape[0])  # width, height
            cv2.grabCut(frame, fgmask, rect, bgdModel, fgdModel, 5,
                        cv2.GC_INIT_WITH_RECT)

            fgmask = np.where((fgmask == 2) | (fgmask == 0), 0,
                              1).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_mog(self, frames):
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
        fgmasks = []
        for frame in tqdm(frames):
            fgmask = fgbg.apply(frame)
            fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_mog2(self, frames):
        fgbg = cv2.createBackgroundSubtractorMOG2()
        fgmasks = []
        for frame in tqdm(frames):
            fgmask = fgbg.apply(frame)
            fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_gsoc(self, frames):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGSOC()
        fgmasks = []
        i = 0
        for frame in tqdm(frames):
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_gmg(self, frames):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
        fgmasks = []
        for frame in tqdm(frames):
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = np.where((fgmask == 255), 1, 0).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    #get motion vector
    def get_foreground_mask_mv(self, frames, bs, k, threshold):
        frame_count, height, width, _ = frames.shape
        frames_yuv = np.array(
            [cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) for frame in frames])
        fgmasks = np.zeros((frame_count, height, width), np.uint8)

        mv = MotionVector()

        for fn in tqdm(range(1, frame_count)):
            for y in range(0, height, bs):
                for x in range(0, width, bs):
                    bw = bs if x + bs <= width else width - x
                    bh = bs if y + bs <= height else height - y
                    dir_y, dir_x = mv.getBlockMV(frames_yuv[fn - 1],
                                                 frames_yuv[fn], y, x, bh, bw,
                                                 k)
                    if dir_y**2 + dir_x**2 > threshold**2:
                        fgmasks[fn, y:y + bh, x:x + bw] = 1
            cv2.imshow("dasd", frames[fn] * fgmasks[fn, :, :, np.newaxis])
            cv2.waitKey(100)

        return fgmasks

    #https://pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
    #https://thedatafrog.com/en/articles/human-detection-video/
    def get_foreground_mask_hog(self, frames):
        # detect people in the image
        fgmasks = []
        for frame in tqdm(frames):
            #rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8) )
            (rects, weights) = self.hog.detectMultiScale(frame,
                                                         winStride=(8, 8),
                                                         padding=(2, 2),
                                                         scale=1.05)
            # draw the original bounding boxes
            for (x, y, w, h) in rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.imshow("lol", frame)
            # cv2.waitKey(500)
            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            fgmask = np.zeros(frame.shape[:2], np.uint8)

            for (xA, yA, xB, yB) in pick:
                # draw the final bounding boxes
                # cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                # apply GrabCut using the the bounding box segmentation method
                # try to cut human off from each box
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                rect = (xA, yA, xB - xA, yB - yA)
                cv2.grabCut(frame, fgmask, rect, bgdModel, fgdModel, 5,
                            cv2.GC_INIT_WITH_RECT)
            fgmask = np.where((fgmask == 2) | (fgmask == 0), 0,
                              1).astype('uint8')
            fgmasks.append(fgmask)
        return np.array(fgmasks)

    def get_foreground_mask_dof(self, frames):
        prvs = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frames[0])
        hsv[..., 1] = 255
        fgmasks = []
        fgmasks.append(np.zeros(frames[0].shape[:2], np.uint8))
        for i in tqdm(range(1, len(frames))):
            next = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            #(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #the last parameter we use OPTFLOW_FARNEBACK_GAUSSIAN = 256
            #the sixth parameter determines the window size, the larger it is the blurer we use 50
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 50,
                                                3, 5, 1.5, 256)
            # flow = self.subsample(flow)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        #     # the motion angles extracted from the vectors are values in 360 degrees
        #     # dividing them by 72 partitions them into 5 segments for finding the background(most common) motion angle
        #     ang = ang / 72
        #     m, n = ang.shape
        #     u, c = np.unique(ang.round(), return_counts=True)
        #     most_common_ang = u[c.argmax()]
        #     mask = np.zeros_like(ang)

        #     # this loop filters out foreground objects by comparing against the most common angle
        #     angle_threshold =0.4
        #     horizontal_edge_buffer = 5
        #     vertical_edge_buffer = 0
        #     for x in range(vertical_edge_buffer, m-vertical_edge_buffer):
        #         for y in range(horizontal_edge_buffer, n-horizontal_edge_buffer):
        #             if min(abs(ang[x][y] - most_common_ang),
        #                 abs(5 - ang[x][y] - most_common_ang)) > angle_threshold:
        #                 mask[x][y] = 1
        #     prvs = next
        #     fgmasks.append(mask) 
        # print("fgmasks", np.array(fgmasks).shape)
        # return np.array(fgmasks)

            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #now we convert every fram to 0 or 255
            (thresh,
             im_bw) = cv2.threshold(bgr[:, :, 2:3], 128, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.imshow('frame2', im_bw)
            k = cv2.waitKey(30) & 0xff
            # replay any > 0 to 1
            im_bw[im_bw > 0] = 1
            prvs = next
            fgmasks.append(im_bw)
        print("fgmasks", np.array(fgmasks).shape)
        return np.array(fgmasks)

    def update_background(self, current_frame, prev_bg, learningRate):
        bg = learningRate * current_frame + (1 - learningRate) * prev_bg
        bg = np.uint8(bg)
        return bg

    def get_foreground_mask_dst(self, frames):
        THRESH = 60
        n = len(frames)
        bg = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        fgmasks = []
        fgmasks.append(np.zeros(frames[0].shape[:2], np.uint8))
        for i in tqdm(range(1, n)):
            #Convert frame to grayscale
            frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            #D(N) = || I(t) - I(t+N) || = || I(t-N) - I(t) ||
            diff = cv2.absdiff(bg, frame_gray)
            #Mask Thresholding
            threshold_method = cv2.THRESH_BINARY
            # when we meet threshold we assign that pixel to 255
            ret, motion_mask = cv2.threshold(diff, THRESH, 255,
                                             threshold_method)
            # Update background at every 1/frames rate
            bg = self.update_background(frame_gray, bg, 0.1)
            #Display the Motion Mask
            cv2.imshow('Motion Mask', motion_mask)
            k = cv2.waitKey(30) & 0xff
            fgmasks.append(motion_mask)
        return np.array(fgmasks)
