import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
         client_id='800dda130a1e4a9d91e2da37bfc1befe',
         client_secret='545434cbe5d64104875c94ebc2e00cc2'
))
def process_data(data):
    data.rename(columns={'album_id':'user_id'}, inplace = True)
    data['track_artist'] = data['artist_name'] + '_' + data['track_name']
    data_cleaned = data.drop_duplicates(subset=['user_id', 'track_artist'])
    return data_cleaned

def find_top_genres(data, number_of_top):
    data['genres'] = data['genres'].str.split()
    exploded_genres = data.explode('genres')
    genre_counts = exploded_genres['genres'].value_counts()
    top_genres = genre_counts.head(number_of_top)
    genre_names_list_10 = top_genres.index.tolist()
    del genre_names_list_10[0]
    return genre_names_list_10

def tfidf_vectorization(data, genre_names_list):
    data['genres_list'] = data['genres'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    genre_set = set(genre_names_list)
    data['filtered_genres'] = data['genres_list'].apply(
        lambda x: [genre for genre in x if genre in genre_set]
    )
    data['filtered_genres_str'] = data['filtered_genres'].apply(
        lambda x: ' '.join(x)
    )
    vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform(data['filtered_genres_str'])
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), columns=[f'genre|feature{i}' for i in range(10)]
    )
    final_data = pd.concat([data.reset_index(drop=True), tfidf_df], axis=1)
    final_data = final_data.drop(
        columns=['genres_list', 'filtered_genres', 'filtered_genres_str']
    )
    return final_data

def scale_data(data_for_scaling, audio_feature_columns_to_scale):

    audio_features_data = data_for_scaling[audio_feature_columns_to_scale]
    data_for_scaling = pd.DataFrame(sc.fit_transform(audio_features_data), columns=audio_feature_columns_to_scale)
    return data_for_scaling
def conc(tdf_all_data, data_for_scaling, audio_feature_columns_to_scale):

    data_for_tdf = tdf_all_data.drop(columns=audio_feature_columns_to_scale)
    final_dataset_all = pd.concat([data_for_tdf.reset_index(drop=True), data_for_scaling.reset_index(drop=True)], axis=1)
    return final_dataset_all


def get_IDs(playlist_url):

    playlist_id = re.search(r'playlist/([a-zA-Z0-9]+)', playlist_url).group(1)
    track_ids = []
    playlist = sp.playlist(playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        track_ids.append(f"spotify:track:{track['id']}")
    return track_ids

def check_url(playlist_url):
    if not playlist_url.startswith("https://open.spotify.com/playlist/"):
        print("Not a valid Spotify playlist URL")
        return []
    else:
        track_ids = get_IDs(playlist_url)
        print("Track IDs:", track_ids)
        return track_ids

def check_input_len(songs_url):
    if len(songs_url) < 5 or len(songs_url) > 250:
        print("Too many or too few songs to process")
        return False
    return True
def check_final_df(dataframe):
    if len(dataframe) < 5:
        print('Too few songs to make recommendation')
        return False
    return True

def filter_dataframe_by_tracks(dataframe, songs_url):
    if not check_input_len(songs_url):
        print("Invalid input length for songs_url.")
        return pd.DataFrame()

    filtered_rows = []
    for uri in songs_url:
        matching_row = dataframe[dataframe['track_uri'] == uri]
        if not matching_row.empty:
            filtered_rows.append(matching_row)

    if filtered_rows:
        filtered_df = pd.concat(filtered_rows, ignore_index=True)
    else:
        filtered_df = pd.DataFrame()

    return filtered_df

def final_preprocessing(final_dataset_all, filtered_df, columns_to_drop_user, columns_to_drop_data):

    columns_to_drop_user = ['artist_name', 'track_uri', 'artist_uri', 'track_name', 'album_uri',
       'album_name', 'user_id', 'release_date', 'genres', 'track_artist']
    columns_to_drop_data = ['artist_name', 'track_uri', 'artist_uri', 'track_name', 'album_uri',
        'album_name', 'release_date', 'genres', 'track_artist']
    final_dataset_all.drop(columns = columns_to_drop_data, inplace=True)
    filtered_df.drop(columns = columns_to_drop_user, inplace=True)

    user_playvec = pd.DataFrame(filtered_df.sum(axis=0)).T
    initial_playvec = final_dataset_all.groupby('user_id').sum(numeric_only=True)
    initial_playvec = initial_playvec.reset_index()

    return user_playvec, initial_playvec

def find_similarity(data, user_playvec, initial_playvec):
    features = initial_playvec.drop(columns=['user_id'])
    playvec_features = user_playvec.drop(columns=['artist_name', 'track_uri', 'artist_uri', 'track_name', 'album_uri',
        'album_name', 'release_date', 'genres', 'user_id', 'track_artist',], errors='ignore')

    common_columns = features.columns.intersection(playvec_features.columns)

    features = features[common_columns]
    playvec_features = playvec_features[common_columns]

    n_samples, n_features = features.shape
    n_components = min(n_samples, n_features) - 1

    pca = PCA(n_components=n_components)
    initial_playvec_pca = pca.fit_transform(features)

    playvec_pca = pca.transform(playvec_features.values.reshape(1, -1))
    similarity_scores = cosine_similarity(playvec_pca, initial_playvec_pca).flatten()

    highest_index = similarity_scores.argmax()
    highest_user_id = initial_playvec.iloc[highest_index]['user_id']
    highest_score = similarity_scores[highest_index]

    print(f'user_id: {highest_user_id}, score: {highest_score}')
    user_songs = data.loc[data['user_id'] == highest_user_id]
    df_sorted = user_songs.sort_values(by='popularity', ascending=False)
    df_filtered = df_sorted[['track_name', 'artist_name', 'track_uri']]
    return df_filtered

### Songs


def find_top_genres(data, number_of_top):
    data = data.copy()
    data['genres'] = data['genres'].apply(
        lambda g: g if isinstance(g, list)
        else g.split() if isinstance(g, str)
        else []
    )
    exploded_genres = data.explode('genres')
    genre_counts = exploded_genres['genres'].value_counts()
    top_genres = genre_counts.head(number_of_top)
    return top_genres

def tfidf_vectorization(data, genre_names_list):
    data = data.copy()
    data['genres'] = data['genres'].apply(
        lambda g: g if isinstance(g, list) 
        else g.split() if isinstance(g, str) 
        else []
    )

    data['filtered_genres'] = data['genres'].apply(
        lambda genre_list: " ".join([genre for genre in genre_list if genre in genre_names_list])
    )

    tfidf = TfidfVectorizer(max_features=10)
    tfidf_matrix = tfidf.fit_transform(data['filtered_genres'])
    genre_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    genre_df.columns = ['genre|' + genre for genre in genre_df.columns]

    data = pd.concat([data.reset_index(drop=True), genre_df.reset_index(drop=True)], axis=1)
    data.drop(columns=['filtered_genres'], inplace=True)
    return data

def compute_recommendations(final_dataset_all, final_playlist_dataset, top_n=50):
    playvec = pd.DataFrame(final_playlist_dataset.sum(axis=0)).T

    drop_cols = ['track_uri', 'artist_uri', 'album_uri', 'artist_name', 'track_name', 'album_name', 'release_date', 'genres', 'album_id']
    df = final_dataset_all.copy()
    df_numeric = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    playvec_numeric = playvec.drop(columns=[col for col in drop_cols if col in playvec.columns], errors='ignore')

    df_numeric = df_numeric.fillna(0)
    playvec_numeric = playvec_numeric.fillna(0)

    df['sim'] = cosine_similarity(df_numeric.iloc[:, 7:19], playvec_numeric.iloc[:, 7:19]).flatten()

    pop_cols = df_numeric.columns[df_numeric.columns.str.startswith('popularity')]
    df['sim2'] = cosine_similarity(df_numeric[pop_cols], playvec_numeric[pop_cols]).flatten()

    pop_autors = df_numeric.columns[df_numeric.columns.str.startswith('artist_pop')]
    df['sim5'] = cosine_similarity(df_numeric[pop_autors], playvec_numeric[pop_autors]).flatten()

    genre_cols = df_numeric.columns[df_numeric.columns.str.startswith('genre|')]
    df['sim3'] = cosine_similarity(df_numeric[genre_cols], playvec_numeric[genre_cols]).flatten()

    df['sim4'] = (df['sim'] +  df['sim2'] * 0.9 + df['sim3'] + df['sim5'] *1.5) / 4

    sorted_recommendations = df.sort_values(by='sim4', ascending=False, kind='stable')
    limited_recommendations = sorted_recommendations.groupby('track_name').head(1)
    limited_recommendations = limited_recommendations.groupby('artist_uri').head(3)

    top_recommendations = limited_recommendations.head(top_n)
    return top_recommendations

def extract_track_uri(input_id):

    match = re.search(r'track/([a-zA-Z0-9]+)', input_id)
    if match:
        raw_id = match.group(1)
        return f"spotify:track:{raw_id}"
    else:
        if 'spotify:track:' in input_id:
            return input_id.strip()
        else:
            raise ValueError("Invalid input: unable to extract a valid track URI.")

def create_recommendations_for_track(input_id):
    train_df = pd.read_csv("./final.csv")

    track_uri = extract_track_uri(input_id)
    if track_uri not in train_df['track_uri'].values:
        raise ValueError("No such song in the dataset")
    playlist_dataset = train_df[train_df['track_uri'] == track_uri].copy()

    top_6_genres = find_top_genres(playlist_dataset, 6)
    genre_names_list = top_6_genres.index.tolist()

    tdif_playlist_dataset = tfidf_vectorization(playlist_dataset, genre_names_list)

    audio_feature_columns = [
        'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'time_signature'
    ]

    data_for_tdf = train_df.copy()
    data_for_scaling = train_df.copy()

    tdf_all_data = tfidf_vectorization(data_for_tdf, genre_names_list)

    sc = MinMaxScaler()
    train_audio_features = data_for_scaling[audio_feature_columns]
    sc.fit(train_audio_features)

    scaled_train_audio = pd.DataFrame(sc.transform(train_audio_features), columns=audio_feature_columns)
    final_dataset_all = tdf_all_data.drop(columns=audio_feature_columns, errors='ignore')
    final_dataset_all = pd.concat([final_dataset_all.reset_index(drop=True),
                                   scaled_train_audio.reset_index(drop=True)], axis=1)

    playlist_audio_features = tdif_playlist_dataset[audio_feature_columns]
    scaled_playlist_audio = pd.DataFrame(sc.transform(playlist_audio_features), columns=audio_feature_columns)
    tdif_playlist_dataset = tdif_playlist_dataset.drop(columns=audio_feature_columns, errors='ignore')
    final_playlist_dataset = pd.concat([tdif_playlist_dataset.reset_index(drop=True),
                                        scaled_playlist_audio.reset_index(drop=True)], axis=1)

    input_track_uris = [track_uri]
    final_dataset_all = final_dataset_all[~final_dataset_all['track_uri'].isin(input_track_uris)]

    top_recommendations = compute_recommendations(final_dataset_all, final_playlist_dataset, top_n=50)
    return top_recommendations[['track_name', 'artist_name', 'track_uri']][1:]
from PIL import Image
def main():

    st.markdown(
    """
    <style>
    .stApp { background-color: #1DB954; color: white; }
    [data-testid="stSidebar"] { background-color: #191414; }
    input, button { border: none; }

    /* Button Styling */
    button {
        background-color: white !important;
        color: #1DB954 !important;
        font-weight: bold;
        border-radius: 5px;
        padding: 8px 16px;
    }
    button:hover {
        background-color: #e6ffe6 !important;
        color: #1DB954 !important;
    }

    /* Slider Styling */
    [data-testid="stSlider"] {
        background-color: white !important;
        border-radius: 10px;
        padding: 5px;
    }
    [data-testid="stSlider"] .css-qbe2hs {
        color: #1DB954 !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] .stRadio label {
        color: white !important; /* Ensure radio labels are white */
    }

    /* Radio Buttons specific styles */
    [data-testid="stSidebar"] .stRadio input + label {
        color: white !important; /* Style radio labels when the button is unselected */
    }

    [data-testid="stSidebar"] .stRadio input:checked + label,
    [data-testid="stSidebar"] .stRadio input:hover + label,
    [data-testid="stSidebar"] .stRadio input:focus + label {
        color: white !important; /* Ensure the selected, hovered, or focused radio label is white */
    }

    </style>
    """,
    unsafe_allow_html=True
)
    img = Image.open('./download.png')
    st.image(img)
    st.title("Music Recommendation System")
    st.sidebar.header("Input Options")

    input_choice = st.sidebar.radio("Choose Input Type", ("Song", "Playlist"))

    if input_choice == "Playlist":
        playlist_url = st.text_input("Enter Spotify Playlist URL")
        num_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

        if st.button("Generate Recommendations"):
            if playlist_url:
                songs_url = check_url(playlist_url)
                if songs_url:
                    data = pd.read_csv('./final.csv')
                    data_cleaned = process_data(data)
                    tdf_all_data = tfidf_vectorization(data_cleaned, find_top_genres(data_cleaned, 11))

                    audio_feature_columns_to_scale = ['duration_ms', 'danceability', 'energy', 'key', 'loudness',
                                                      'mode', 'speechiness', 'acousticness', 'instrumentalness',
                                                      'liveness', 'valence', 'tempo', 'time_signature', 'popularity',
                                                      'artist_pop']
                    data_for_scaling = scale_data(tdf_all_data, audio_feature_columns_to_scale)
                    final_dataset_all = conc(tdf_all_data, data_for_scaling, audio_feature_columns_to_scale)

                    filtered_df = filter_dataframe_by_tracks(final_dataset_all, songs_url)
                    if check_final_df(filtered_df):
                        columns_to_drop_user = ['artist_name', 'track_uri', 'artist_uri', 'track_name', 'album_uri',
                                                'album_name', 'user_id', 'release_date', 'genres', 'track_artist']
                        columns_to_drop_data = ['artist_name', 'track_uri', 'artist_uri', 'track_name', 'album_uri',
                                                'album_name', 'release_date', 'genres', 'track_artist']
                        user_playvec, initial_playvec = final_preprocessing(final_dataset_all, filtered_df,
                                                                            columns_to_drop_user, columns_to_drop_data)
                        user_songs = find_similarity(data, user_playvec, initial_playvec)

                        st.write("### Recommended Songs:")
                        st.write(user_songs.head(num_recommendations))
                    else:
                        st.error("Too few songs in the playlist to make a recommendation.")
                else:
                    st.error("Invalid Spotify playlist URL.")
            else:
                st.error("Please enter a Spotify playlist URL.")

    elif input_choice == "Song":

        song_url = st.text_input("Enter Spotify Song URL")
        num_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

        if st.button("Generate Recommendations"):
            if song_url:
                try:
                    recommendations = create_recommendations_for_track(song_url)
                    if not recommendations.empty:
                        st.write("### Recommended Songs:")
                        st.write(recommendations.head(num_recommendations))
                    else:
                        st.error("No recommendations found for the provided song URL.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Please enter a Spotify song URL.")


if __name__ == '__main__':
    main()
