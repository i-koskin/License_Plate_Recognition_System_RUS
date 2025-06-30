import cv2
from pathlib import Path
from datetime import datetime


def create_video_writer(frame_shape, source_label):
    h = int(frame_shape[0])
    w = int(frame_shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{now}_{source_label}.avi"
    output_path = Path("recordings")
    output_path.mkdir(exist_ok=True)

    return cv2.VideoWriter(str(output_path / filename), fourcc, 25.0, (w, h))
