from ultralytics import YOLO
import supervision as spv
import pickle
import os 
import cv2
import sys
sys.path.append('../')
import numpy as np
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = spv.ByteTrack()
    
    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch
        return detections
    
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        # For each frame, and its detections
        for frame_idx, detection in enumerate(detections):
            cls_names = detection.names
            #reverse the dictionary, easier to lookup the class name
            cls_names_inv = {v:k for k,v in cls_names.items()} 
            
            #Convert the detections to the supervision lib format
            detection_spv =spv.Detections.from_ultralytics(detection)
            
            #Convert goalkeepers to players, due to the limitation of the training library
            for obj_idx, class_id in enumerate(detection_spv.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_spv.class_id[obj_idx] = cls_names_inv['player']
                
            #Tracking the objects
            detection_w_tracks = self.tracker.update_with_detections(detection_spv)
            
            #Append each tracked object to the tracks dictionary 
            tracks["players"].append({})        
            tracks["referees"].append({})        
            tracks["ball"].append({})
            
            for frame_detection in detection_w_tracks:
                bbox = frame_detection[0].tolist()
                cls_id =  frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_names[cls_id] == 'player':         
                    tracks['players'][frame_idx][track_id] = {"bbox": bbox}
                
                elif cls_names[cls_id] == 'referee':
                    tracks['referees'][frame_idx][track_id] = {"bbox": bbox}
                
                for frame_detection in detection_spv:
                    bbox = frame_detection[0].tolist()
                    cls_id =  frame_detection[3]
                    #only 1 ball, no need for a track id
                    
                    if cls_names[cls_id] == 'ball':
                        tracks['ball'][frame_idx][1] = {"bbox": bbox}
            
        
        if stub_path is not None and read_from_stub is False:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                    
        return tracks        
    
    #draw an ellipse around the player, at the bottom of the bounding box
    def draw_ellipse(self, frame, bbox, color, track_id=None, team_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        frame = cv2.ellipse(frame,
                            center=(x_center, y2),
                            axes=(int(width), int(0.35*width)),
                            angle=0,
                            startAngle=-45,
                            endAngle=235,
                            color=color,
                            thickness=1,
                            lineType=cv2.LINE_4) 
        
        #draw a rectangle below the player showing the player number
        #TODO: add player speed tracking
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = y2 - rectangle_height//2 + 15
        y2_rect = y2 + rectangle_height//2 + 15
    
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color, 
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -=  10
            
            label_text = f"{track_id}"
            if team_id is not None:
                label_text += f" Team {team_id}"
              
            cv2.putText(frame,
                        label_text,
                        (int(x1_text), int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2)
        return frame
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        #assume the ball would fit in an inverted isosceles triangle with a base of 20 pixels
        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y+20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        #draw a black border
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), cv2.FILLED)
        
    def draw_annotations(self, frames, tracks):
        output_frames = []
        frames_copy = frames.copy()
        for frame_idx, frame in enumerate(frames_copy):
            player_dict = tracks['players'][frame_idx]
            referee_dict = tracks['referees'][frame_idx]
            ball_dict = tracks['ball'][frame_idx]
            
            #Drawing players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id, player.get('team'))
            
            #Drawing referees
            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))
            
            #Drawing ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (255, 0, 255))
                
            output_frames.append(frame)
            
        return output_frames
    
    
    