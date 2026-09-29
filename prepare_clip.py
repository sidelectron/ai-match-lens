"""Extract a short, silent clip for the first calibration run."""
import argparse
from pathlib import Path
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('video')
parser.add_argument('--start', type=float, default=74)
parser.add_argument('--seconds', type=float, default=5)
parser.add_argument('--output', default='football_clip.mp4')
args = parser.parse_args()
if args.start < 0 or args.seconds <= 0:
    parser.error('Start must be nonnegative and duration must be positive.')
target = Path(args.output)
if target.exists():
    parser.error(f'{target} already exists; choose a different --output.')
cap = cv2.VideoCapture(args.video)
writer = None
try:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise RuntimeError('Could not open the video.')
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)
    count = 0
    for _ in range(round(args.seconds * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError('Could not create the clip.')
            cv2.imwrite(str(target.with_suffix('.jpg')), frame)
        writer.write(frame)
        count += 1
    if not count:
        raise RuntimeError('No frames available at that start time.')
    print(f'Saved {count / fps:.1f} seconds to {target.resolve()}')
finally:
    cap.release()
    if writer is not None:
        writer.release()
