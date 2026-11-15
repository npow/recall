"""Contact sheets of the presentation video (one frame every N seconds) for a visual check."""
import sys

import cv2
import numpy as np


def main():
    video, prefix = sys.argv[1], sys.argv[2]
    step = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, t = [], step / 2
    while t * fps < total:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = capture.read()
        if ok:
            frame = cv2.resize(frame, (960, 540))
            cv2.putText(frame, f"{t:.1f}s", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            frames.append(frame)
        t += step
    for sheet in range(0, len(frames), 6):
        chunk = frames[sheet:sheet + 6]
        while len(chunk) < 6:
            chunk.append(np.zeros_like(frames[0]))
        grid = np.vstack([np.hstack(chunk[0:2]), np.hstack(chunk[2:4]), np.hstack(chunk[4:6])])
        path = f"{prefix}-{sheet // 6:02d}.jpg"
        cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(path)
    print(total / fps, "seconds")


if __name__ == "__main__":
    main()
