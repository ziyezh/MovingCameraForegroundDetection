import cv2
import numpy as np
from tqdm import tqdm

from panorama.foreground_extraction import ForegroundExtractor
from panorama.matcher import matcher


class Video:
    FG_GRABCUT = "grabcut"
    FG_MOG = "mog"
    FG_MOG2 = "mog2"
    FG_GSOC = "gsoc"
    FG_GMG = "gmg"
    FG_HOG = "hog"
    FG_DOF = "dof"
    FG_LKO = "lko"
    FG_MV = "mv"  # motion vector
    FG_DST = "dst"

    def __init__(self, filepath: str) -> None:
        self._cap = cv2.VideoCapture(filepath)
        self._background = np.zeros(shape=[self.width, self.height, 3],
                                    dtype=np.uint8)
        self.filename = filepath.split('/')[-1].split('.')[0]
        self._frames = np.array([])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._cap.release()

    def set_background(self, background: np.ndarray) -> None:
        self._background = background

    def mergeForegroundManual(
        self,
        bg: np.ndarray,
        fg: np.ndarray,
        checkpoint_interval: int = 4,
        n: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        print('merge panorama and foreground...')
        print(
            'Manual match: please click the four corners of the foreground by the order of top-left, bottom-left, bottom-right, top-right.'
        )
        w, h = fg[0].shape[1], fg[0].shape[0]
        fg_pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        prev: tuple[int, any] = [-1, np.zeros(shape=[3, 3], dtype=np.float64)]
        frames = []
        out1 = bg.copy()
        transformations = []

        frame_checkpoints = list(
            range(0, fg.shape[0],
                  self.fps * checkpoint_interval)) + [fg.shape[0] - 1]

        for i in frame_checkpoints:
            img = bg.copy()
            positions: list[list[int]] = []

            def draw_circle(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONUP:
                    cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
                    positions.append([x, y])

            cv2.namedWindow('image')
            cv2.setMouseCallback('image', draw_circle)
            while len(positions) < 4:
                cv2.imshow('image', img)
                cv2.imshow('fg', self.frames[i])
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow('image')
            cv2.destroyWindow('fg')

            H, mask = cv2.findHomography(
                fg_pts,
                np.float32(positions).reshape(-1, 1, 2), cv2.RANSAC, 5.0)

            count = i - prev[0]
            step = (H - prev[1]) / count
            curH = prev[1]

            for j in range(count):
                curH += step
                transformations.append(curH)
                reg = cv2.warpPerspective(fg[prev[0] + 1 + j], curH,
                                          (bg.shape[1], bg.shape[0]))
                frame = self.overlay_image_alpha(bg, reg)
                frames.append(frame)

                if (prev[0] + 1 + j) % (self.fps * n) == 0:
                    out1 = self.overlay_image_alpha(out1, reg)

            prev = (i, H)
        return np.array(frames), out1, np.array(transformations)

    def mergeForeground(self,
                        bg: np.ndarray,
                        fg: np.ndarray,
                        n: int = 1) -> tuple[np.ndarray, np.ndarray]:
        print('merge panorama and foreground...')
        frames = []
        out1 = bg.copy()
        m = matcher()
        transformations = []

        for i in tqdm(range(fg.shape[0])):
            H = m.match(bg, self.frames[i])
            if H is None:
                frames.append(self.frames[-1])
                transformations.append(transformations[-1])
                continue

            transformations.append(H)
            h, w = bg.shape[0], bg.shape[1]
            fgReg = cv2.warpPerspective(fg[i], H, (w, h))
            frame = self.overlay_image_alpha(bg, fgReg)
            frames.append(frame)

            if i % (self.fps * n) == 0:
                out1 = self.overlay_image_alpha(out1, fgReg)
        return np.array(frames), out1, np.array(transformations)

    def createNewCamera(
        self,
        bg: np.ndarray,
        frames: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        dimension=(640, 480)) -> list[np.ndarray]:
        start = self._normalize_coordinates(
            *start,
            *dimension,
            bg.shape[1],
            bg.shape[0],
        )
        end = self._normalize_coordinates(
            *end,
            *dimension,
            bg.shape[1],
            bg.shape[0],
        )
        dx = (end[0] - start[0]) / frames.shape[0]
        dy = (end[1] - start[1]) / frames.shape[0]
        halfWidth = int(0.5 * dimension[0])
        halfHeight = int(0.5 * dimension[1])

        new_frames = []
        camera_center: list[float] = [start[0], start[1]]

        for i in tqdm(range(frames.shape[0])):
            frame = frames[i]
            lx, rx = int(camera_center[0] - halfWidth), int(camera_center[0] +
                                                            halfWidth)
            ly, ry = int(camera_center[1] - halfHeight), int(camera_center[1] +
                                                             halfHeight)

            new_frames.append(frame[ly:ry, lx:rx])
            camera_center[0] += dx
            camera_center[1] += dy
        return new_frames

    def _normalize_coordinates(self, x: int, y: int, cameraWidth: int,
                               cameraHeight: int, bgWidth: int,
                               bgHeight: int) -> tuple[int, int]:
        return (
            max(cameraWidth // 2, min(bgWidth - cameraWidth // 2, x)),
            max(cameraHeight // 2, min(bgHeight - cameraHeight // 2, y)),
        )

    def write(self, filename: str, frames: list[np.ndarray] | np.ndarray,
              w: int, h: int) -> None:
        file = cv2.VideoWriter(f'{filename}.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                               (w, h))
        for frame in frames:
            file.write(frame)
        file.release()

    @property
    def fps(self) -> int:
        # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return int(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frames(self) -> np.ndarray:
        if len(self._frames) > 0:
            return self._frames

        frames = []
        while (self._cap.isOpened()):
            ret, frame = self._cap.read()
            if ret is True:
                frames.append(frame)
            else:
                break
        self._frames = np.array(frames)

        return self._frames

    def extract_foreground(
            self, mode: str,
            config: any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        print("Extracting foreground...")
        fgmasks = []
        extractor = ForegroundExtractor()
        frames = self.frames

        if mode == Video.FG_GRABCUT:
            fgmasks = extractor.get_foreground_mask_grabcut(frames)
        elif mode == Video.FG_MOG:
            fgmasks = extractor.get_foreground_mask_mog(frames)
        elif mode == Video.FG_MOG2:
            fgmasks = extractor.get_foreground_mask_mog2(frames)
        elif mode == Video.FG_GSOC:
            fgmasks = extractor.get_foreground_mask_gsoc(frames)
        elif mode == Video.FG_GMG:
            fgmasks = extractor.get_foreground_mask_gmg(frames)
        elif mode == Video.FG_HOG:
            fgmasks = extractor.get_foreground_mask_hog(frames)
        elif mode == Video.FG_DOF:
            fgmasks = extractor.get_foreground_mask_dof(frames)
        elif mode == Video.FG_MV:
            fgmasks = extractor.get_foreground_mask_mv(
                frames, int(config.mv_blocksize), int(config.mv_k),
                float(config.mv_threshold))
        elif mode == Video.FG_DST:
            fgmasks = extractor.get_foreground_mask_dst(frames)
        else:
            raise Exception("Invalid fgmode")

        bgmasks = np.where((fgmasks == 1), 0, 1).astype('uint8')
        # print(frames.shape, fgmasks.shape, fgmasks[:, :, :, np.newaxis].shape)

        fg = frames * fgmasks[:, :, :, np.newaxis]
        bg = frames * bgmasks[:, :, :, np.newaxis]

        return fg, bg, fgmasks

    def show(self, frames: np.ndarray) -> None:

        for frame in frames:
            cv2.imshow('frame', frame)
            # & 0xFF is required for a 64-bit system
            if cv2.waitKey(1000 // self.fps) & 0xFF == ord('q'):
                break

    def overlay_image_alpha(
            self,
            img: np.ndarray,
            overlay: np.ndarray,
            bgLowerBound=np.array([0, 0, 0]),
            bgUpperBound=np.array([5, 5, 5]),
    ) -> np.ndarray:
        mask = cv2.inRange(overlay, bgLowerBound, bgUpperBound)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        return cv2.bitwise_or(overlay, masked_img)
