from utils import read_video, save_video
from trackers import Tracker

def main():
    frames = read_video('in_videos/08fd33_4.mp4')
    
    #Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames, True, 'stubs/08fd33_4_tracks.pkl')
    
    #Draw annotations on the output video frames
    out_frames = tracker.draw_annotations(frames, tracks)
    
    save_video(out_frames, 'out_videos/08fd33_4_out.avi')
    
if __name__ == '__main__':
    main()