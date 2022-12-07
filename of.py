import numpy as np
import cv2 as cv


def read_RGB_file(name):
    file = open(name, "r")
    # to 3-d arr
    BGR = np.fromfile(file, np.uint8)
    BGR = BGR.reshape(height, width, 3)
    # BGR to RGB
    BGR[:, :, [2, 0]] = BGR[:, :, [0, 2]]
    RGB = BGR
    return RGB

blockSize = 16
def subsample(arr):
    row = arr.shape[0]
    col = arr.shape[1]
    depth = arr.shape[2]
    n_row = (row - 1) // blockSize + 1
    n_col = (col - 1) // blockSize + 1
    n = np.zeros((n_row, n_col, depth))
    for i in range(n_row):
        for j in range(n_col):
            for k in range(depth):
                n[i][j][k] = arr[i * blockSize:(i + 1) * blockSize:, j * blockSize:(j + 1) * blockSize:, k].mean()
    return n

def bgrToGray(each_frame):
    return cv.cvtColor(each_frame, cv.COLOR_BGR2GRAY)

paras = ["video1_240_424_518", 0.4, 15, 0, 8, 5, 0]
folder_name = paras[0]
angle_threshold = paras[1]
nearest_frames = paras[2]
quantile_threshold = paras[3]
min_shape_threshold = paras[4]
horizontal_edge_buffer = paras[5]
vertical_edge_buffer = paras[6]

Images = []

w_h_fps = folder_name.split("_")
width = int(w_h_fps[1])
height = int(w_h_fps[2])
frames = int(w_h_fps[3])
path = folder_name + "/"
for i in range(frames):
    filename = folder_name + "." + "{:03d}".format(i + 1) + ".rgb"
    name = path + filename
    Images.append(read_RGB_file(name))

frame1 = Images[0]
prvs = bgrToGray(frame1)

foreground_frames = []
background_frames = []
frame_masks = []
# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html dense optical flow
for i in range(1, len(Images)):
    nxt = bgrToGray(Images[i])
    # motion vectors
    # def calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags):
    # motion = cv.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    motion = cv.calcOpticalFlowPyrLK(prvs, nxt, None, 0.5, 3, 16, 3, 5, 0.5, 1)
    subsampled_motion = subsample(motion)
    subsampled_mag, subsampled_ang = cv.cartToPolar(subsampled_motion[..., 0], subsampled_motion[..., 1],
                                                    angleInDegrees=True)
    # the motion angles extracted from the vectors are values in 360 degrees
    # dividing them by 72 partitions them into 5 segments for finding the background(most common) motion angle
    subsampled_ang = subsampled_ang / 72
    m, n = subsampled_ang.shape
    u, c = np.unique(subsampled_ang.round(), return_counts=True)
    most_common_ang = u[c.argmax()]
    mask = np.zeros_like(subsampled_ang)

    # this loop filters out foreground objects by comparing against the most common angle
    for x in range(vertical_edge_buffer, m-vertical_edge_buffer):
        for y in range(horizontal_edge_buffer, n-horizontal_edge_buffer):
            if min(abs(subsampled_ang[x][y] - most_common_ang),
                   abs(5 - subsampled_ang[x][y] - most_common_ang)) > angle_threshold:
                mask[x][y] = 1
    frame_masks.append(mask)
    prvs = nxt

foreground_video_output = cv.VideoWriter("output_farneback/" + folder_name + '/foreground_video.mp4',cv.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

for i in range(len(Images)):
    # print(i)
    left_frame = 0
    right_frame = i + nearest_frames
    if i - nearest_frames > left_frame :
        left_frame = i - nearest_frames
    # mean in each col
    matrix = np.mean(frame_masks[left_frame:right_frame], axis=0)
    total_shapes = 0
    shapes = {}
    m, n = frame_masks[i].shape
    # this loop uses bfs to find all contiguous shapes and discards small moving segments
    for x in range(m):
        # print("x")
        # print(x)
        for y in range(n):
            # print("y")
            # print(y)
            if matrix[x][y] == 0:
                continue
            stack = [(x, y)]
            area = 0
            coord = set()
            visited = set()
            while stack:
                u, v = stack.pop()
                if 0 <= u < m and 0 <= v < n and (u, v) not in visited:
                    visited.add((u, v))
                    if matrix[u][v] > quantile_threshold:
                        area += 1
                        coord.add((u, v))
                        stack.append((u - 1, v))
                        stack.append((u + 1, v))
                        stack.append((u, v - 1))
            if area > 0:
                if len(coord) > 2:
                    shapes[total_shapes] = coord
                    total_shapes += 1

    # this step upscales the foreground mask and produces the foreground object frame and the background frame with objected removed
    foreground_frame = np.full_like(Images[i], 255)
    background_frame = Images[i].copy()
    shapes_dict = {}
    for shape in shapes.values():
        if len(shape) > min_shape_threshold:
            for x, y in shape:
                shapes_dict[(x, y)] = shape
                foreground_frame[x * blockSize:(x + 1) * blockSize, y * blockSize:(y + 1) * blockSize] = Images[i][x * blockSize:(x + 1) * blockSize,
                                                                                     y * blockSize:(y + 1) * blockSize]
                background_frame[x * blockSize:(x + 1) * blockSize, y * blockSize:(y + 1) * blockSize] = [0, 0, 0]

    foreground_frames.append(foreground_frame)
    background_frames.append(background_frame)
    filename = "output_farneback/" + folder_name + "/background/" + str(i) + ".jpg"
    cv.imwrite(filename, background_frame)

    # cv.imshow('frame', foreground_frame)
    # cv.setWindowProperty('frame', cv.WND_PROP_TOPMOST, 1)
    k = cv.waitKey(30) & 0xff
    # print(k)
    if k == 27:
        break
    foreground_video_output.write(foreground_frame)


print("-------------over-----------")

foreground_video_output.release()
cv.destroyAllWindows()