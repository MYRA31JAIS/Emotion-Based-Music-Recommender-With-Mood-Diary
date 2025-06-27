# Emotion-based music recommendation system

This is a project that detects real-time facial emotions using a CNN model and recommends music accordingly via Spotify integration.
It also maintains a timestamped emotional diary with poetic entries, offering both personalization and sentiment tracking.
Built using Python, TensorFlow, OpenCV, and Streamlit.

 Features:

- 🎥 Detects facial emotions using your webcam in real-time
- 🤖 Predicts emotion using a trained CNN model (FER-2013)
- 🎵 Recommends music from an emotion-tagged dataset (MUSE)
- 🎧 Spotify API integration for direct music playback
- ❤️ Feedback system (Like/Dislike) for smarter future suggestions
- ⭐ Save your favorite tracks and revisit them anytime
- 📓 My Diary: Auto-generates mood-based entries with poetic vibes
- 💅 Built with a glowing pink-themed UI using Streamlit + CSS


🛠️ Tech Stack

| Layer               | Tools & Libraries                             |
|---------------------|-----------------------------------------------|
| Frontend            | Streamlit, HTML, CSS                          |
| Backend             | Python                                        |
| Face Detection      | OpenCV, Haarcascade                           |
| Emotion Prediction  | Keras, TensorFlow (CNN trained on FER-2013)   |
| Music Integration   | Spotify API (`spotipy`)                       |
| Dataset             | MUSE (`muse_v3.csv`)                          |
| Data Handling       | Pandas, NumPy                                 |
| Storage             | JSON (`favorites.json`, `user_feedback.json`, `my_diary.json`) |

Installation & Setup

1.Clone the Repository:

2.Install Dependencies:

```bash
  pip install -r requirements.txt
```

3.Run the Application:

  streamlit run app.py
  
4.How to Use

-Click "SCAN EMOTION" to capture live emotion using webcam.
-The system detects your emotion and shows 5 matching songs.
-You can ❤️ Like, 👎 Dislike, or ⭐ Save songs to your favorites.
-View saved tracks in the 💖 Favorites tab.
-Check and manage your mood diary in the 📓 My Diary section.

📌 Future Enhancements:

📱 Mobile Application Version
🎧 Personalized User Playlists
🌍 Multilingual Support
🤖 AI Chatbot for Music Suggestions

📜 License:

This project is open-source under the MIT License.

❤️ Contributions:

Feel free to fork this repository, contribute, and submit pull requests!

## Authors

- [Myra Jaiswal]
