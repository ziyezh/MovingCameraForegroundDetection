import cv2
import numpy as np
from tqdm import tqdm

from panorama.matcher import matcher


class FillBackGround:

    def mse(self, img1, img2):
        #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        height, width, _ = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff**2)
        mse = err / (float(height * width))
        return mse

    #using BFS to get foreground and background
    def fillBackground(self, bg, fgmasks):
        print("Filling background...")
        frame_count, height, width, channel = bg.shape
        Threshold = 10
        B, G, R, count = 0, 0, 0, 0
        for a in tqdm(range(frame_count)):
            #cv2.imwrite("./tmp/frame%d.jpg" % a, backGround[a])
            for b in range(frame_count):
                #if the missing pixel is found in other frame then we can fill the missing one
                m = mse(bg[a], bg[b])
                if a != b and m < Threshold:
                    for i in range(height):
                        for j in range(width):
                            #fgmask = 0 means background
                            sa = bg[a][i][j][0] + bg[a][i][j][1] + bg[a][i][j][
                                2]
                            sb = bg[b][i][j][0] + bg[b][i][j][1] + bg[b][i][j][
                                2]
                            if fgmasks[a][i][j] and fgmasks[b][i][
                                    j] == 0 and sa == 0 and sb:
                                bg[a][i][j][0] = bg[b][i][j][0]
                                bg[a][i][j][1] = bg[b][i][j][1]
                                bg[a][i][j][2] = bg[b][i][j][2]
                                fgmasks[a][i][j] = 0
                            elif fgmasks[a][i][j] == 0 and fgmasks[b][i][
                                    j] and sb == 0 and sa:
                                bg[b][i][j][0] = bg[a][i][j][0]
                                bg[b][i][j][1] = bg[a][i][j][1]
                                bg[b][i][j][2] = bg[a][i][j][2]
                                fgmasks[b][i][j] = 0
            for i in range(height):
                for j in range(width):
                    #if the pixel is still 0 then we use average of its surrounding pixel to fill it
                    if fgmasks[a][i][j]:
                        I = i - 1
                        while I > -1 and fgmasks[a][I][j]:
                            I -= 1
                        if I > -1:
                            B += bg[a][I][j][0]
                            G += bg[a][I][j][1]
                            R += bg[a][I][j][2]
                            count += 1
                        I = i + 1
                        while I < height and fgmasks[a][I][j]:
                            I += 1
                        if I < height:
                            B += bg[a][I][j][0]
                            G += bg[a][I][j][1]
                            R += bg[a][I][j][2]
                            count += 1
                        J = j - 1
                        while J > -1 and fgmasks[a][i][J]:
                            J -= 1
                        if J > -1:
                            B += bg[a][i][J][0]
                            G += bg[a][i][J][1]
                            R += bg[a][i][J][2]
                            count += 1
                        J = j + 1
                        while J < width and fgmasks[a][i][J]:
                            J += 1
                        if J < width:
                            B += bg[a][i][J][0]
                            G += bg[a][i][J][1]
                            R += bg[a][i][J][2]
                            count += 1

                        if count:
                            bg[a][i][j][0] = B / count
                            bg[a][i][j][1] = G / count
                            bg[a][i][j][2] = R / count
                        else:
                            print("maybe go diagnoal direction ???")

    def fill_background(self, bg, fgmasks, fps):
        print('Go fill background ...')
        n, height, width, channel = bg.shape
        match = matcher()
        sampleBG = []
        # F1=H×F2
        for i in tqdm(range(1, n, 10)):
            H = match.match(bg[i - 1], bg[i], 'right')
            # print(H)
            filled = False
            for j in range(height):
                for k in range(width):
                    #fgmask = 1 means foreground
                    if fgmasks[i][j][k]:
                        correspond = np.dot(H, [j, k, 1])
                        # print('-----------------')
                        # print(correspond)
                        x = min(max(int(correspond[0]), 0), height - 1)
                        y = min(max(int(correspond[1]), 0), width - 1)
                        if fgmasks[i - 1][x][y] == 0:
                            filled = True
                            bg[i][j][k][0] = bg[i - 1][x][y][0]
                            bg[i][j][k][1] = bg[i - 1][x][y][1]
                            bg[i][j][k][2] = bg[i - 1][x][y][2]
            if filled: sampleBG.append(bg[i])
        return sampleBG
