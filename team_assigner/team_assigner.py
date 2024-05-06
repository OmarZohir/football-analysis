from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self) -> None:
        self.team_colors = {}
        self.player_team_dict = {}
    
    
    def get_clustering_model(self, image):
        #Reshape the image into a 2d array
        image_2d = image.reshape(-1, 3)

        #perform k-means with 2 clusters
        kmeans_model = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)
        
        return kmeans_model
        
    def get_player_color(self, frame, bbox):
        
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[:image.shape[0]//2, :]
        
        #get clustering model
        kmeans = self.get_clustering_model(top_half_image)
        
        #get the cluster for each pixel
        labels = kmeans.labels_
        
        #reshape the labels into the original image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # Assume the class for the background will be the one that appears the most on the corners of the image
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 -non_player_cluster
        
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color
        
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans= KMeans(n_clusters=2, init="k-means++", n_init=1).fit(player_colors)
        
        self.kmeans = kmeans
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        # # Perform k-means with 3 clusters
        # kmeans = KMeans(n_clusters=3, init="k-means++", n_init=1).fit(player_colors)
        
        # # Get the labels of the clusters
        # labels = kmeans.labels_
        
        # # Count the frequency of each cluster
        # unique, counts = np.unique(labels, return_counts=True)
        # cluster_counts = dict(zip(unique, counts))
        
        # # Sort the clusters by frequency and select the two most frequent
        # sorted_clusters = sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True)
        # team_clusters = sorted_clusters[:2]
        
        # # Assign the cluster centers of the two most frequent clusters to the teams
        # self.team_colors[1] = kmeans.cluster_centers_[team_clusters[0][0]]
        # self.team_colors[2] = kmeans.cluster_centers_[team_clusters[1][0]]
        
        # self.kmeans = kmeans
        
    # assign a player to a team
    def get_player_team(self, frame, bbox, player_id):
        # if player_id in self.player_team_dict:
        #     return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        # team 1 and 2, instead of team 0 and 1 (labelling)
        team_id+=1
        
        #add the player to the team dict
        self.player_team_dict[player_id] = team_id
        
        return team_id
        
        