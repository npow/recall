"""frames_strip.py OUT.jpg RUN_DIR [RUN_DIR ...]: one row per run, frames at 0, 1/3, 2/3 and the end of the concatenated chunks."""
from pathlib import Path
import sys

import cv2
import numpy as np

rows = []
for run in sys.argv[2:]:
    frames = []
    for path in sorted(Path(run).glob("chunk-*.mp4")):
        cap = cv2.VideoCapture(str(path))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    picks = [frames[int(i * (len(frames) - 1) / 3)] for i in range(4)]
    picks = [cv2.resize(f, (320, 184)) for f in picks]
    label = Path(run).name.rsplit("-", 1)[0]
    cv2.putText(picks[0], label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    rows.append(np.hstack(picks))
cv2.imwrite(sys.argv[1], np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
