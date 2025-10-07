import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

@st.cache_data
def load_movies():
    df = pd.read_csv('tmdb_5000_movies.csv')
    df['overview'] = df['overview'].fillna('')
    return df

moviesList = load_movies()

@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    return cosine_similarity(tfidf_matrix)

similarity = compute_similarity(moviesList)

@st.cache_data
def compute_scores(df):
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.90)
    df['score'] = df.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) +
                                      (m/(x['vote_count']+m) * C), axis=1)
    return df

movie_ratings = compute_scores(moviesList.copy())

def recommend(movie):
    ind = moviesList[moviesList['title'] == movie].index[0]
    distances = similarity[ind]
    recommended_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [moviesList.iloc[i[0]].title for i in recommended_list]

def recommenderFnVoteBased(genre):
    genre_clean = genre.strip().lower()
    recommendedList = []
    ct = 0

    for idx, row in movie_ratings.iterrows():
        g_raw = row['genres']
        genres_list = []

        # Handle stringified lists
        if isinstance(g_raw, str):
            try:
                parsed = ast.literal_eval(g_raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and 'name' in item:
                            genres_list.append(item['name'].strip().lower())
                        else:
                            genres_list.append(str(item).strip().lower())
                else:
                    genres_list = [str(parsed).strip().lower()]
            except:
                genres_list = [g.strip().lower() for g in g_raw.split(" ")]
        # Already a list
        elif isinstance(g_raw, list):
            for item in g_raw:
                if isinstance(item, dict) and 'name' in item:
                    genres_list.append(item['name'].strip().lower())
                else:
                    genres_list.append(str(item).strip().lower())
        else:
            genres_list = [str(g_raw).strip().lower()]

        if genre_clean in genres_list:
            recommendedList.append({'title': row['title'], 'score': round(row['score'], 2)})
            ct += 1
            if ct == 3:
                break

    if not recommendedList:
        return f'No movies found for the genre: "{genre}"'
    return recommendedList

st.title('🎬 Movie Recommender System')
tab1, tab2 = st.tabs(["By Movie", "By Genre"])

#Content Based
with tab1:
    st.header('Recommendation based on selected Movie')
    selectedMovie = st.selectbox('Select Movie', moviesList['title'].values)
    recommendations = []
    if st.button('Recommend based on Movie'):
        recommendations = recommend(selectedMovie)

    if len(recommendations) != 0:
        st.subheader(f'Movies similar to "{selectedMovie}":')
        for i in recommendations:
            st.write(i)

# Genre Based
with tab2:
    st.header('Recommendation based on Genre')

    # Collect all genres dynamically and safely
    movie_genre = set()
    for g_raw in movie_ratings['genres'].dropna():
        genres_list = []

        if isinstance(g_raw, str):
            try:
                parsed = ast.literal_eval(g_raw)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and 'name' in item:
                            genres_list.append(item['name'].strip())
                        else:
                            genres_list.append(str(item).strip())
                else:
                    genres_list = [str(parsed).strip()]
            except:
                genres_list = [g.strip() for g in g_raw.split(" ")]
        elif isinstance(g_raw, list):
            for item in g_raw:
                if isinstance(item, dict) and 'name' in item:
                    genres_list.append(item['name'].strip())
                else:
                    genres_list.append(str(item).strip())
        else:
            genres_list = [str(g_raw).strip()]

        for genre in genres_list:
            movie_genre.add(genre)

    movie_genre = sorted(list(movie_genre))

    selectedGenre = st.selectbox('Select Genre', movie_genre)
    recommendations = []
    if st.button('Recommend based on Genre'):
        recommendations = recommenderFnVoteBased(selectedGenre)

    if len(recommendations) != 0 and isinstance(recommendations, list):
        st.subheader(f'Top "{selectedGenre}" Movies:')
        for i in recommendations:
            st.write(i['title'])
    elif isinstance(recommendations, str):
        st.write(recommendations)
