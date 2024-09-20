import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os

# Spotify authentication
auth_manager = SpotifyClientCredentials(client_id='a0b4db73b6c049de9efa7afbd9783db9', client_secret='ae5fc1a2e1e04aa0b371dbb1c3a4b631')
sp = spotipy.Spotify(auth_manager=auth_manager)

def getTrackIDs(playlist_id):
    track_ids = []
    playlist = sp.playlist_tracks(playlist_id)
    for item in playlist['items']:
        track = item['track']
        if track and track['id']:
            track_ids.append(track['id'])
    return track_ids

def getTrackFeatures(id):
    if id is None:
        return None
    track_info = sp.track(id)
    name = track_info['name']
    album = track_info['album']['name']
    artist = track_info['album']['artists'][0]['name']
    track_data = [name, album, artist]
    return track_data

# Emotion dictionaries
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
music_dist = {
    0: "0KPEhXA3O9jHFtpd1Ix5OB",  # Updated playlist for Angry
    1: "7gS8udDbiuPUXjBqZ9bOUr",  # Disgusted
    2: "7gS8udDbiuPUXjBqZ9bOUr",  # Fearful
    3: "7gS8udDbiuPUXjBqZ9bOUr",  # Happy
    4: "7gS8udDbiuPUXjBqZ9bOUr",  # Neutral
    5: "7gS8udDbiuPUXjBqZ9bOUr",  # Sad
    6: "7gS8udDbiuPUXjBqZ9bOUr"   # Surprised
}

# Create 'songs' directory if it does not exist
os.makedirs('songs', exist_ok=True)

for emotion in emotion_dict.keys():
    track_ids = getTrackIDs(music_dist[emotion])
    track_list = []
    for i in range(len(track_ids)):
        time.sleep(0.3)
        track_data = getTrackFeatures(track_ids[i])
        if track_data:
            track_list.append(track_data)
    df = pd.DataFrame(track_list, columns=['Name', 'Album', 'Artist'])
    df.to_csv(f'songs/{emotion_dict[emotion].lower()}.csv')
    print(f"CSV for {emotion_dict[emotion]} Generated")
