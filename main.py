from utils import read_video, save_video
from trackers import Tracker
import os
import cv2
from team_assigner import TeamAssigner

#assume video extensions are .mp4 only, can be automated tho
def process_single_video(video_file, read_from_stub=False):
    frames = read_video(f'in_videos/{video_file}.mp4')
    
    #Initialize tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(frames, read_from_stub, f'stubs/{video_file}_tracks.pkl')
    
    
    #Assign players to teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team_id = team_assigner.get_player_team(frames[frame_num],
                                                    track['bbox'],
                                                    player_id)
             
            tracks['players'][frame_num][player_id]['team'] = team_id
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_id]
            
        
    #Draw annotations on the output video frames
    out_frames = tracker.draw_annotations(frames, tracks)
    
    save_video(out_frames, f'out_videos/{video_file}_out.avi')
    

def process_all_videos(read_from_stubs=False):
    video_files = os.listdir('in_videos')
    for video_file in video_files:
        if not video_file.endswith('.mp4'):
            continue
        frames = read_video(f'in_videos/{video_file}.mp4')
        
        #Initialize tracker
        tracker = Tracker('models/best.pt')
        tracks = tracker.get_object_tracks(frames, read_from_stubs, f'stubs/{video_file}_tracks.pkl')
        
        #Assign players to teams
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(frames[0],
                                        tracks['players'][0])
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team_id = team_assigner.get_player_team(frames[frame_num],
                                                        track['bbox'],
                                                        player_id)
                
                tracks['players'][frame_num][player_id]['team'] = team_id
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team_id]
        
        #Draw annotations on the output video frames
        out_frames = tracker.draw_annotations(frames, tracks)
        
        save_video(out_frames, f'out_videos/{video_file}_out.avi')


def main():
    process_all_videos()
    
if __name__ == '__main__':
    main()