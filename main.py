import argparse
import os
import glob

import cv2
import numpy as np

from panorama.object_remover import ObjectRemover
from panorama.fill_background import FillBackGround
from panorama.StitchPanorama import StitchPanorama
from panorama.video import Video
from panorama.draw_line import DrawLineWidget

panoramas = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", required=True)
    parser.add_argument("-fg", "--fgmode", default=Video.FG_MOG2)
    parser.add_argument("-bs", "--mv_blocksize", default=16)
    parser.add_argument("-k", "--mv_k", default=16)
    parser.add_argument("-th", "--mv_threshold", default=15)
    parser.add_argument("-c",
                        "--clear",
                        action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("-a",
                        "--automatic",
                        action=argparse.BooleanOptionalAction,
                        default=False)

    parser.add_argument("-dw", "--width", default=640)
    parser.add_argument("-dh", "--height", default=480)

    return parser.parse_args()


def get_video_cache(filename: str) -> np.ndarray:
    cap = cv2.VideoCapture(filename)
    frames = []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret is True:
            frames.append(frame)
        else:
            break
    frames = np.array(frames)
    return frames


def main(config: argparse.Namespace) -> None:
    with Video(config.filepath) as cap:
        if config.clear:
            print('Clearing file cache...')
            for f in glob.glob(f"{cap.filename}_*"):
                os.remove(f)

        fg, bg, fgmasks = cap.extract_foreground(config.fgmode, config)
        cap.write(f'{cap.filename}_fg', fg, cap.width, cap.height)
        # fg = get_video_cache(f'{cap.filename}_fg.mp4')

        panoFile = f'{cap.filename}_pano.jpg'
        if not os.path.exists(panoFile):
            # remove foreground and fill out the removed part in background
            # this issue involved camera motion, size change, object tracking
            fbg = FillBackGround()
            sampleBG = fbg.fill_background(bg, fgmasks, cap.fps)
            # using processed background and stitch them together to create panorama
            # we just need to sample 5 points for stitching Q1 - Q5
            sp = StitchPanorama(sampleBG)
            cv2.imwrite(panoFile, sp.simpleStitch())
        else:
            print('Cached panorama file is used.')

        pano = cv2.imread(panoFile)
        res, out1, h = cap.mergeForegroundManual(
            pano, fg) if not config.automatic else cap.mergeForeground(
                pano, fg)
        cv2.imwrite(f'{cap.filename}_out1.jpg', out1)
        cap.write(f'{cap.filename}_result', res, pano.shape[1], pano.shape[0])

        res = get_video_cache(f'{cap.filename}_result.mp4')
        # print(
        #     'Draw a line to indicate the direction of camera motion and press q to leave'
        # )
        # camera = DrawLineWidget(pano, res)
        # while True:
        #     cv2.imshow(camera.window_name, camera.show_image())
        #     key = cv2.waitKey(1)
        #     if key == ord('q'):
        #         cv2.destroyWindow(camera.window_name)
        #         break
        # out2 = cap.createNewCamera(pano, res, camera.image_coordinates[0],
        #                            camera.image_coordinates[1],
        #                            (config.width, config.height))
        print(pano.shape)
        out2 = cap.createNewCamera(pano, res, (0, pano.shape[0] / 2), (pano.shape[1], pano.shape[0] / 2), (config.width, config.height))
        cap.write(f'{cap.filename}_out2', out2, config.width, config.height)

        print("Creating output3...")
        obj_remover = ObjectRemover()
        fg_removed = obj_remover.remove_largest_object(fg)
        bgmasks_removed = np.where((fg_removed < 5), 1, 0).astype('uint8')
        bg_removed = bg * bgmasks_removed
        out3 = bg_removed + fg_removed
        cap.write(f'{cap.filename}_out3', out3, cap.width, cap.height)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
