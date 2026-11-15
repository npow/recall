"""Contact strip of two videos (top / bottom) for choosing presentation clips."""
import sys

import cv2
import numpy as np


def frames(path, count=9, width=240):
    capture = cv2.VideoCapture(path)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for index in np.linspace(0, max(total - 1, 0), count).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            continue
        height = int(frame.shape[0] * width / frame.shape[1])
        out.append(cv2.resize(frame, (width, height)))
    capture.release()
    return np.hstack(out), total


def main():
    output, top, bottom = sys.argv[1:4]
    a, na = frames(top)
    b, nb = frames(bottom)
    width = min(a.shape[1], b.shape[1])
    cv2.imwrite(output, np.vstack([a[:, :width], b[:, :width]]))
    print(output, na, nb)


if __name__ == "__main__":
    main()
