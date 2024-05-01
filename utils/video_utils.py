import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

#Read the video frame by frame, as a generator
def read_video_gen(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

def save_video(out_frames, out_video_path, fps=24):
    if not out_frames:
        print("INFO: No frames to save.")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #assume all frames are of the same size
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (out_frames[0].shape[1], out_frames[0].shape[0]))
    for frame in out_frames:
        out.write(frame)
    out.release()
    
    
    