# ✅ Enhanced Emotion-Based Music Recommendation System with Favorites & My Diary

import numpy as np
import streamlit as st
import cv2
import pandas as pd
import random
import os
import json
from dotenv import load_dotenv
load_dotenv()
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime

# 🚀 Caching model and data loading for performance
@st.cache_resource
def load_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights("model.h5")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("muse_v3.csv")
    df['link'] = df['lastfm_url']
    df['name'] = df['track']
    df['emotional'] = df['number_of_emotion_tags']
    df['pleasant'] = df['valence_tags']
    df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
    df = df.sort_values(by=["emotional", "pleasant"])
    df.reset_index(drop=True, inplace=True)
    return (
        df.iloc[:18000],
        df.iloc[18000:36000],
        df.iloc[36000:54000],
        df.iloc[54000:72000],
        df.iloc[72000:]
    )

model = load_model()
df_sad, df_fear, df_angry, df_neutral, df_happy = load_data()
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_mapping = {
    'Neutral': df_neutral,
    'Angry': df_angry,
    'Fearful': df_fear,
    'Happy': df_happy,
    'Sad': df_sad,
    'Surprised':df_happy,
    'Disgusted':df_sad}

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Spotify setup
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(auth_manager=auth_manager)


def get_spotify_link(song_name, artist_name):
    try:
        query = f"track:{song_name} artist:{artist_name}"
        print(f"[DEBUG] Searching Spotify for: {query}")
        results = spotify.search(q=query, type="track", limit=1)

        if results['tracks']['items']:
            link = results['tracks']['items'][0]['external_urls']['spotify']
            print(f"[DEBUG] Found Spotify link: {link}")
            return link
        else:
            print(f"[DEBUG] No results found for: {query}")
    except Exception as e:
        print(f"[ERROR] Spotify search failed: {e}")
    return None


# Feedback system
FEEDBACK_FILE = "user_feedback.json"
FAV_FILE = "favorites.json"
DIARY_FILE = "my_diary.json"

def load_json(file):
    return json.load(open(file)) if os.path.exists(file) else {}

def save_json(file, data):
    try:
        with open(file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[DEBUG] Saved JSON to {file}")
    except Exception as e:
        st.error(f"Error saving {file}: {e}")
        
def load_json(file):
    return json.load(open(file)) if os.path.exists(file) else {}


user_feedback = load_json(FEEDBACK_FILE)
favorites = load_json(FAV_FILE)
diary = load_json(DIARY_FILE)

def update_feedback(song_name, liked):
    user_feedback[song_name] = liked
    save_json(FEEDBACK_FILE, user_feedback)

def add_to_favorites(song_name, artist, link):
    # Fallback if link is empty or NaN
    if not link or pd.isna(link):
        link = "https://open.spotify.com/"  # default fallback link

    # Avoid duplicate entries
    if song_name not in favorites:
        favorites[song_name] = {
            "artist": artist,
            "link": link
        }
        save_json(FAV_FILE, favorites)
        st.toast(f"⭐ '{song_name}' added to favorites!")
        print(f"[DEBUG] Added to favorites: {song_name} by {artist} | Link: {link}")
    else:
        print(f"[DEBUG] '{song_name}' already in favorites.")
        
        for i, (l, a, n) in enumerate(zip(data["link"], data['artist'], data['name']), 1):
                spotify_link = get_spotify_link(n, a) or l
                st.markdown(f"<h4><a href='{spotify_link}' target='_blank'> 💜 {i}. {n} </a></h4>", unsafe_allow_html=True)
                st.markdown(f"<h5>✨ {a} ✨</h5>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                if c1.button(f"❤️ Like {i}"):
                    update_feedback(n, 1)
                if c2.button(f"👎 Dislike {i}"):
                    update_feedback(n, -1)
                if c3.button(f"⭐ Save {i}"):
                    add_to_favorites(n, a, spotify_link)
def add_to_diary(emotion):
    today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story = generate_diary_entry(emotion)
    diary[today] = story
    save_json(DIARY_FILE, diary)

def generate_diary_entry(emotion):
    shayaris = {
        "Happy": [
            "Dil garden garden ho gaya, vibes full on fun hai! 🎉🌈",
            "Zindagi ka mood on point, sab kuch lag raha hai lit! 🔥😄",
            "Smile aayi bina reason ke, lagta hai universe bhi chill hai! ✨😊"
        ],
        "Sad": [
            "Dil halka sa udaas hai, par memes toh ab bhi funny hain... 😢📱",
            "Aaj coffee bhi thodi extra bitter lagi... ya mood ka glitch tha? ☕💭",
            "Soft songs + warm blanket = therapy mode activated. 🎧🛏️"
        ],
        "Angry": [
            "Gussa aaya full power, par chill bhi toh ek superpower hai! ⚡😤",
            "Mood: roasted like my daily toast. 🍞🔥",
            "Thoda patience, thoda playlist — control + alt + delete kar diya! 🎵🧘"
        ],
        "Fearful": [
            "Dil ne bola 'uh-oh', dimaag ne bola 'you got this!' 💡🫣",
            "Nervous thoda sa hoon, par hero wali entry toh banta hai! 🎬🦸",
            "Scared? Yes. Still showing up? Absolutely! 😎✨"
        ],
        "Neutral": [
            "Na zyada udaan, na heavy down — bas ek chill sa vibe hai. 😌☁️",
            "Mood today: do not disturb, but not upset either. 🚪🧘",
            "Jo ho raha hai, theek hi ho raha hai — aaj ka status yehi hai. 📶🌀"
        ]
    }
    return random.choice(shayaris.get(emotion, ["Bas ek ajeeb sa vibe hai, par manageable hai. 🤷"]))


def pre(emotion_list):
    if not emotion_list:
        return ["Neutral"]
    emotion_counts = Counter(emotion_list)
    dominant_emotion, count = emotion_counts.most_common(1)[0]
    return [dominant_emotion] if count >= len(emotion_list) * 0.4 else ["Neutral"]

def refine_recommendations(data):
    liked = [song for song in data["name"] if user_feedback.get(song) == 1]
    disliked = [song for song in data["name"] if user_feedback.get(song) == -1]
    
    # Remove disliked songs
    refined = data[~data["name"].isin(disliked)]

    # Move liked songs to top (without duplication)
    liked_df = refined[refined["name"].isin(liked)]
    refined = refined[~refined["name"].isin(liked)]
    refined = pd.concat([liked_df, refined]).drop_duplicates(subset="name")

#ensures 5 recommendations atleast
    return refined.sample(n=min(5, len(refined))) if not refined.empty else data.sample(n=5)


st.markdown("""
    <style>
    body { background-color: #0a0a0a; color: white; }
    h2, h5 { text-align: center; color: #ff69b4; font-family: 'Poppins', sans-serif; text-shadow: 1px 1px 5px #ff69b4; }
    .stButton > button { background-color: #ff1493; color: white; font-weight: bold; border-radius: 25px; padding: 10px 20px; box-shadow: 0px 0px 15px #ff69b4; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2><b>✨ Emotion-Based Music Recommendation 🎶</b></h2>", unsafe_allow_html=True)

menu = st.sidebar.radio("📚 Menu", ["🎧 Scan & Recommend", "💖 My Favorites", "📝 My Diary"])

if menu == "🎧 Scan & Recommend":
    st.markdown("<h2><b>✨ Scan your face to get started 🎥</b></h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    emotion_list = []

    if col2.button("✨ SCAN EMOTION ✨"):
        cap = cv2.VideoCapture(0)
        stframe = st.image([])
        count = 0

        while count < 15:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.copyMakeBorder(frame, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(120, 81, 169))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img, verbose=0)
                detected_emotion = emotion_dict[np.argmax(prediction)]
                emotion_list.append(detected_emotion)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            count += 1

        cap.release()
        cv2.destroyAllWindows()

        dominant_emotion = pre(emotion_list)[0]
        st.success(f"✨ Detected Emotion: {dominant_emotion} ✨")
        st.text(f"🧠 Emotion Summary: {Counter(emotion_list)}")

        add_to_diary(dominant_emotion)

        if dominant_emotion in emotion_mapping:
            st.markdown("<h5>💜 Recommended Songs 💕</h5>", unsafe_allow_html=True)
            data = emotion_mapping[dominant_emotion].sample(n=5)
            data = refine_recommendations(data)
           
            for i, (l, a, n) in enumerate(zip(data["link"], data['artist'], data['name']), 1):
                spotify_link = get_spotify_link(n, a) 
                if not spotify_link or pd.isna(spotify_link):
                   spotify_link = l if pd.notna(l) else "https://open.spotify.com/"

                st.markdown(f"<h4><a href='{spotify_link}' target='_blank'> 💜 {i}. {n} </a></h4>", unsafe_allow_html=True)
                st.markdown(f"<h5>✨ {a} ✨</h5>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                if c1.button(f"❤️ Like {i}", key=f"like_{i}_{n}"):
                    update_feedback(n, 1)
                    st.toast(f"❤️ You liked: {n}")
                if c2.button(f"👎 Dislike {i}", key=f"dislike_{i}_{n}"):
                    update_feedback(n, -1)
                    st.toast(f"👎 You disliked: {n}")
                if c3.button(f"⭐ Save {i}", key=f"save_{i}_{n}"):
                    spotify_link = get_spotify_link(n, a) or 1
                    add_to_favorites(n, a, spotify_link)
                    st.toast(f"⭐ Added to favorites: {n}")
elif menu == "💖 My Favorites":
    st.markdown("<h2><b>💖 Your Favorite Tracks</b></h2>", unsafe_allow_html=True)

    if favorites:
        to_delete = []

        for i, (name, info) in enumerate(favorites.items(), 1):

            link = info.get('link')
            if not link or pd.isna(link):
                link = "https://open.spotify.com/"

            artist = info.get('artist', 'Unknown Artist')

            # ✅ Now you can safely display the song and artist
            st.markdown(f"<h4><a href='{link}' target='_blank'> 💜 {i}. {name} </a></h4>", unsafe_allow_html=True)
            st.markdown(f"<h5>✨ {artist} ✨</h5>", unsafe_allow_html=True)

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🗑️ Remove", key=f"remove_{i}"):
                    to_delete.append(name)

        if to_delete:
            for song in to_delete:
                favorites.pop(song, None)
            save_json(FAV_FILE, favorites)
            st.success("✅ Selected song(s) removed from favorites.")
            st.experimental_rerun()

    else:
        st.info("You haven’t added any songs to favorites yet.")

elif menu == "📝 My Diary":
    st.markdown("<h2><b>📓 My Emotion Diary</b></h2>", unsafe_allow_html=True)

    # Always show the clear diary option
    with st.expander("🧹 Clear Diary"):
        st.warning("This will permanently delete all diary entries.")
        if st.button("Yes, clear my diary"):
            diary.clear()
            save_json(DIARY_FILE, diary)
            st.success("✨ Diary cleared successfully!")

    if diary:
        for date, entry in reversed(list(diary.items())):
            st.markdown(f"**🕒 {date}**")
            st.markdown(f"<pre style='background:#111; color:#ffb6c1; padding:10px'>{entry}</pre>", unsafe_allow_html=True)
    else:
        st.info("Diary is empty. Start scanning your emotion!")

