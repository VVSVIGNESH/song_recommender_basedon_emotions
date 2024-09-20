# song_recommender_basedon_emotions

## Music Recommendation-system using emotion detection
Dataset is downloaded from https://www.kaggle.com/msambare/fer2013
In the same folder, create a new folder named data and save the test and train folders in it. (This repository already has the dataset download and saved in it).
## Packages need to be installed
Run pip install -r requirements.txt to install all dependencies.
## To train Emotion detector model
Run Train.py
After Training , you will find the trained model structure and weights are stored in your project directory. emotion_model.json and emotion_model.h5.
Copy these two files, create model folder in your project directory and paste it. (The pre-trained model is available in the model folder in this repository).
## To run your emotion detection file
Run detect_emotion.py
You can either take your live camera feed or paste the path of the video by commenting the code other out in Test.py
## To analyse the model
Run Evaluate.py
## To run the webapp
Run app.py and visit the link http://127.0.0.1:5000/

## Features
- Real-time emotion detection using a webcam.
- Song recommendations based on detected emotions.
- Simple and user-friendly web interface.

## Technologies Used
- Python
  - TensorFlow for deep learning model.
  - OpenCV for webcam and image processing.
  - NumPy for numerical operations.
  - Spotipy for Spotify API interactions.
  - Pandas for handling CSV files.
  - Flask for the web application.
- HTML, CSS, and JavaScript for the frontend.

### Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
