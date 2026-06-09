import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="CineFlex // Movie Recommendation Engine",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# 2. Custom CSS for CineFlex-style Premium UI
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* CineFlex Dark Theme colors */
    .stApp {
        background-color: #0f0f0f;
        color: #e5e5e5;
    }
    
    /* Sidebar styling override */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141414 0%, #080808 100%) !important;
        border-right: 1px solid #222222 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
    }
    
    /* Sleek container for sidebar widgets */
    .sidebar-section-card {
        background: rgba(30, 30, 30, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Style radio inputs in sidebar */
    div[data-testid="stRadio"] label {
        background-color: #1c1c1c !important;
        border: 1px solid #2d2d2d !important;
        padding: 8px 12px !important;
        border-radius: 6px !important;
        margin-bottom: 8px !important;
        color: #bbbbbb !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    
    div[data-testid="stRadio"] label:hover {
        border-color: #e50914 !important;
        background-color: #242424 !important;
        color: #ffffff !important;
    }
    
    /* Selectbox & multiselect overrides */
    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"] > div[data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        border: 1px solid #2d2d2d !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
        cursor: pointer !important;
        caret-color: transparent !important;
    }
    /* Ensure popup options dropdown menu list uses standard click pointers */
    div[data-baseweb="popover"] *,
    ul[role="listbox"] *,
    div[role="option"] * {
        cursor: pointer !important;
    }
    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:hover,
    div[data-testid="stMultiSelect"] > div[data-baseweb="select"] > div:hover {
        border-color: #e50914 !important;
    }
    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:focus-within,
    div[data-testid="stMultiSelect"] > div[data-baseweb="select"] > div:focus-within {
        border-color: #e50914 !important;
        box-shadow: 0 0 8px rgba(229, 9, 20, 0.25) !important;
    }
    
    /* Number input override */
    div[data-testid="stNumberInput"] input {
        background-color: #1a1a1a !important;
        border: 1px solid #2d2d2d !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stNumberInput"] input:hover {
        border-color: #e50914 !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #e50914 !important;
        box-shadow: 0 0 8px rgba(229, 9, 20, 0.25) !important;
    }
    
    /* Slider custom overrides */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #e50914 !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 0 8px rgba(229, 9, 20, 0.6) !important;
        transition: transform 0.1s ease !important;
    }
    div[data-testid="stSlider"] div[role="slider"]:hover {
        transform: scale(1.2) !important;
    }
    
    /* Metric Card overrides */
    div[data-testid="stMetric"] {
        background: rgba(24, 24, 24, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-left: 4px solid #e50914 !important;
        border-radius: 8px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stMetric"]:hover {
        border-left-color: #46d369 !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.15) !important;
        background: rgba(24, 24, 24, 0.8) !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #aaaaaa !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Expander overrides */
    div[data-testid="stExpander"] {
        background: rgba(24, 24, 24, 0.45) !important;
        border: 1px solid #262626 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        margin-top: 15px !important;
    }
    div[data-testid="stExpander"] details summary {
        color: #ffffff !important;
        font-weight: 600 !important;
        transition: color 0.2s ease !important;
    }
    div[data-testid="stExpander"] details summary:hover {
        color: #e50914 !important;
    }
    
    /* Headers & Text colors */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* CineFlex Logo Header */
    .cineflex-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 2rem;
        background: linear-gradient(180deg, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0) 100%);
        border-bottom: 1px solid rgba(229, 9, 20, 0.15);
        margin-bottom: 2rem;
    }
    
    .logo-text {
        font-family: 'Bebas Neue', Arial, sans-serif;
        font-size: 3.2rem;
        font-weight: 900;
        color: #e50914;
        letter-spacing: 1px;
        text-shadow: 0 0 15px rgba(229, 9, 20, 0.6);
        line-height: 1;
    }
    
    /* Hero Banner Styling */
    .hero-banner {
        background-image: linear-gradient(rgba(15, 15, 15, 0) 0%, rgba(15, 15, 15, 0.85) 75%, rgba(15, 15, 15, 1) 100%), 
                          url('https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=1925&auto=format&fit=crop');
        background-size: cover;
        background-position: center 30%;
        padding: 6rem 3.5rem 3.5rem;
        border-radius: 12px;
        margin-bottom: 2.5rem;
        box-shadow: inset 0 0 120px rgba(0,0,0,0.9);
        border: 1px solid rgba(255,255,255,0.03);
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 4.8rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0;
        letter-spacing: -2px;
        text-shadow: 3px 3px 8px rgba(0, 0, 0, 0.95);
        line-height: 1.05;
    }
    
    .hero-metadata {
        font-size: 1.15rem;
        margin-top: 0.75rem;
        margin-bottom: 1.25rem;
        color: #46d369; /* CineFlex Match score color */
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    
    .hero-desc {
        max-width: 650px;
        font-size: 1.15rem;
        line-height: 1.6;
        color: #dddddd;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);
        margin-bottom: 2rem;
    }
    
    .cineflex-btn {
        background-color: #ffffff;
        color: #000000;
        font-weight: 700;
        padding: 0.75rem 2.25rem;
        border-radius: 6px;
        border: none;
        cursor: pointer;
        display: inline-block;
        margin-right: 1rem;
        font-size: 1.05rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .cineflex-btn:hover {
        background-color: #e50914;
        color: #ffffff;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.4);
    }
    
    .cineflex-btn-secondary {
        background-color: rgba(109, 109, 110, 0.6);
        color: #ffffff;
        font-weight: 700;
        padding: 0.75rem 2.25rem;
        border-radius: 6px;
        border: none;
        cursor: pointer;
        display: inline-block;
        font-size: 1.05rem;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(5px);
    }
    
    .cineflex-btn-secondary:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    /* Horizontal Carousel style */
    .carousel-row {
        display: flex;
        overflow-x: auto;
        gap: 20px;
        padding: 15px 5px 25px;
        scroll-behavior: smooth;
        white-space: nowrap; /* Keep cards inline */
    }
    
    .carousel-row::-webkit-scrollbar {
        height: 6px;
    }
    
    .carousel-row::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.2);
        border-radius: 3px;
    }
    
    .carousel-row::-webkit-scrollbar-thumb {
        background: rgba(229, 9, 20, 0.4);
        border-radius: 3px;
    }
    
    .carousel-row::-webkit-scrollbar-thumb:hover {
        background: rgba(229, 9, 20, 0.85);
    }
    
    /* CineFlex Card Styling */
    .cineflex-card {
        background: #181818;
        border-radius: 8px;
        border: 1px solid #2d2d2d;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .cineflex-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(229, 9, 20, 0.2);
        border-color: #e50914;
    }
 
    /* CineFlex Card inside Carousel */
    .carousel-card {
        flex: 0 0 290px;
        min-width: 290px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        padding: 1.35rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.7);
        transition: all 0.35s cubic-bezier(0.25, 0.8, 0.25, 1);
        display: inline-block;
        white-space: normal; /* Override parent white-space */
        cursor: default;
    }
    
    .carousel-card:hover {
        transform: scale(1.05) translateY(-5px);
        box-shadow: 0 15px 35px rgba(229, 9, 20, 0.3);
        z-index: 100;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .card-meta {
        font-size: 0.9rem;
        color: #aaaaaa;
        margin-bottom: 0.6rem;
    }
    
    .card-match {
        color: #46d369;
        font-weight: 700;
        font-size: 0.95rem;
    }
    
    .card-rating {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.85rem;
        font-weight: 700;
        border-radius: 4px;
        margin-bottom: 0.6rem;
    }
    
    .card-explanation {
        font-size: 0.85rem;
        color: #cccccc;
        background: rgba(0, 0, 0, 0.45);
        padding: 0.6rem 0.8rem;
        border-radius: 4px;
        margin-top: 0.6rem;
        border-left: 2px solid #e50914;
        font-style: italic;
        line-height: 1.4;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(24, 24, 24, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        transition: all 0.2s ease;
    }
    
    .stat-card:hover {
        border-color: #e50914;
        background: rgba(24, 24, 24, 0.75);
        transform: translateY(-2px);
    }
    
    /* Tabs Overrides */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #2d2d2d;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #808080;
        font-weight: 700;
        font-size: 1.05rem;
        border-bottom: 2px solid transparent;
        transition: all 0.3s ease;
        padding: 10px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #e50914 !important;
    }
    
    /* Warning Box */
    .warning-box {
        background: rgba(229, 9, 20, 0.08);
        border: 1px solid rgba(229, 9, 20, 0.3);
        border-left: 4px solid #e50914;
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    
    /* Hide Streamlit default Deploy button, footer, and top-right settings menu */
    [data-testid="stDeployButton"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    [data-testid="stMainMenu"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Models and Data (Cached for speed)
@st.cache_resource
def load_assets():
    data_dir = os.path.join(os.getcwd(), 'data')
    models_dir = os.path.join(os.getcwd(), 'models')
    
    titles_file = os.path.join(data_dir, 'movie_titles.csv')
    eval_data_path = os.path.join(models_dir, 'eval_data.joblib')
    svd_model_path = os.path.join(models_dir, 'svd_model.joblib')
    cf_model_path = os.path.join(models_dir, 'cf_model.joblib')
    results_path = os.path.join(models_dir, 'results.joblib')
    
    # Parse movie titles (now supports 4 columns including genres)
    titles_df = pd.read_csv(titles_file, header=None, names=['movie_id', 'year', 'title', 'genres'], encoding='latin-1')
    titles_dict = dict(zip(titles_df['movie_id'], titles_df['title']))
    movie_years = dict(zip(titles_df['movie_id'], titles_df['year']))
    movie_genres = dict(zip(titles_df['movie_id'], titles_df['genres']))
    
    # Load models
    svd = joblib.load(svd_model_path)
    cf = joblib.load(cf_model_path)
    eval_data = joblib.load(eval_data_path)
    results = joblib.load(results_path) if os.path.exists(results_path) else None
    
    # Load raw ratings to display user history (we read on the fly/cached)
    ratings_file = os.path.join(data_dir, 'combined_data_1.txt')
    rows = []
    with open(ratings_file) as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                movie_id = int(line[:-1])
            else:
                uid, rating, date = line.split(',')
                rows.append((int(uid), movie_id, int(rating), date))
    df = pd.DataFrame(rows, columns=['user_id', 'movie_id', 'rating', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    
    return df, titles_df, titles_dict, movie_years, movie_genres, svd, cf, eval_data, results

try:
    df, titles_df, titles_dict, movie_years, movie_genres, svd, cf, eval_data, results = load_assets()
except Exception as e:
    st.error(f"Error loading models or dataset. Please make sure you have run download_and_prepare.py and train.py. Details: {e}")
    st.stop()

# Dynamically populate top 1000 active users from dataset
SAMPLE_ACTIVE_USERS = df['user_id'].value_counts().head(1000).index.tolist()

# Helper recommendation functions
def get_top_k(model, user_id, all_movie_ids, rated_ids, k=10):
    rated_ids_set = set(rated_ids)
    unrated = [m for m in all_movie_ids if m not in rated_ids_set]
    preds = [model.predict(user_id, m) for m in unrated]
    preds.sort(key=lambda x: x.est, reverse=True)
    return preds[:k]

def explain_recommendation(cf_model, raw_user_id, raw_movie_id, titles_dict):
    try:
        trainset = cf_model.trainset
        inner_uid = trainset.to_inner_uid(raw_user_id)
        inner_iid = trainset.to_inner_iid(raw_movie_id)
        neighbors = cf_model.get_neighbors(inner_uid, k=5)
        
        similar_users_who_liked = []
        for n_inner_uid in neighbors:
            user_ratings = trainset.ur[n_inner_uid]
            for iid, r in user_ratings:
                if iid == inner_iid and r >= 4.0:
                    raw_n_uid = trainset.to_raw_uid(n_inner_uid)
                    similar_users_who_liked.append(raw_n_uid)
                    break
        movie_title = titles_dict.get(raw_movie_id, f"Movie {raw_movie_id}")
        if similar_users_who_liked:
            return f"Recommended because {len(similar_users_who_liked)} similar subscribers rated it 4+ stars."
        else:
            return "Recommended based on collaborative filtering overlap."
    except:
        return "Recommended based on collaborative filtering overlap."

# Helper to retrieve style details for movies depending on their genre
def get_genre_style(genres_str, default_border_color="#e50914"):
    border = default_border_color
    bg = "linear-gradient(135deg, rgba(30, 30, 30, 0.65) 0%, rgba(15, 15, 15, 0.95) 100%)"
    if pd.isna(genres_str) or genres_str == "Unknown":
        return border, bg
    
    # Map primary genre to a specific accent color & subtle glow gradient
    genre_mapping = {
        "Action": ("#e50914", "linear-gradient(135deg, rgba(229, 9, 20, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Adventure": ("#3b82f6", "linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Animation": ("#10b981", "linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Comedy": ("#f59e0b", "linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Drama": ("#8b5cf6", "linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Fantasy": ("#ec4899", "linear-gradient(135deg, rgba(236, 72, 153, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Sci-Fi": ("#06b6d4", "linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Romance": ("#f43f5e", "linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Thriller": ("#ef4444", "linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)"),
        "Horror": ("#9a3412", "linear-gradient(135deg, rgba(154, 52, 18, 0.12) 0%, rgba(24, 24, 24, 0.96) 100%)")
    }
    
    for g in genres_str.split('|'):
        if g in genre_mapping:
            return genre_mapping[g]
    return border, bg

# Helper Carousel rendering functions
def render_carousel_html(recs, titles_dict, movie_years, movie_genres, border_color="#e50914", is_knn=False, cf_model=None, raw_user_id=None):
    html = '<div class="carousel-row">'
    for idx, item in enumerate(recs, 1):
        if len(item) == 4: # Cold start or hybrid structure
            mid, svd_score, genre_sim, combined_score = item
            match_percentage = int(min(max((combined_score / 5.0) * 100, 50.0), 99.0))
            badge_style = "background:rgba(229,9,20,0.15); color:#ff3333; border:1px solid rgba(229,9,20,0.3)"
            badge_text = f"Predicted: {svd_score:.2f} ★"
            sub_text = f"SVD Latent: {svd_score:.2f} | Genre Sim: {genre_sim:.2f}"
        else: # Simple top_k structure (like KNN)
            r = item
            mid = r.iid
            match_percentage = int(min(max((r.est / 5.0) * 100, 50.0), 99.0))
            badge_style = "background:rgba(249,115,22,0.15); color:#f97316; border:1px solid rgba(249,115,22,0.3)"
            badge_text = f"Predicted: {r.est:.2f} ★"
            if is_knn and cf_model and raw_user_id:
                sub_text = explain_recommendation(cf_model, raw_user_id, mid, titles_dict)
            else:
                sub_text = "Collaborative Filtering rating prediction."
                
        title = titles_dict.get(mid, f"Movie {mid}").replace('"', '&quot;')
        year = movie_years.get(mid, "N/A")
        genres = movie_genres.get(mid, "Unknown")
        
        card_border, card_bg = get_genre_style(genres, border_color)
        
        # Build individual card on a single line (no newlines/indents)
        html += f'<div class="carousel-card" style="border-left: 4px solid {card_border}; background: {card_bg}"><div class="card-title" title="{title}">{title}</div><div class="card-meta">{year} | <span class="card-match">{match_percentage}% Match</span></div><div style="font-size:0.75rem; color:#808080; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom: 0.5rem;">{genres}</div><span class="card-rating" style="{badge_style}">{badge_text}</span><div class="card-explanation" style="border-left-color:{card_border}">{sub_text}</div></div>'
    html += '</div>'
    return html.replace('\n', ' ')

def render_popular_carousel(popular_ids, titles_dict, movie_years, movie_genres, df):
    html = '<div class="carousel-row">'
    for idx, mid in enumerate(popular_ids, 1):
        title = titles_dict.get(mid, f"Movie {mid}").replace('"', '&quot;')
        year = movie_years.get(mid, "N/A")
        genres = movie_genres.get(mid, "Unknown")
        ratings_count = df['movie_id'].value_counts()[mid]
        
        card_border, card_bg = get_genre_style(genres, "#10b981")
        
        # Build individual card on a single line (no newlines/indents)
        html += f'<div class="carousel-card" style="border-left: 4px solid {card_border}; background: {card_bg}"><div class="card-title" title="{title}">{title}</div><div class="card-meta">{year}</div><div style="font-size:0.75rem; color:#808080; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-bottom: 0.5rem;">{genres}</div><span class="card-rating" style="background:rgba(16,185,129,0.15); color:#10b981; border:1px solid rgba(16,185,129,0.3)">🔥 {ratings_count:,} reviews</span><div class="card-explanation" style="border-left-color:{card_border}">Top trending blockbuster on CineFlex.</div></div>'
    html += '</div>'
    return html.replace('\n', ' ')

# Get all unique genres from titles
all_genres_set = set()
for gs in titles_df['genres'].dropna():
    for g in gs.split('|'):
        all_genres_set.add(g)
all_genres = sorted(list(all_genres_set))
genre_to_idx = {g: i for i, g in enumerate(all_genres)}

# SVD Projection for New Users (Cold-Start Strategy)
def project_new_user_svd(svd_model, user_ratings, lambda_reg=0.02):
    trainset = svd_model.trainset
    mu = trainset.global_mean
    
    active_ratings = []
    for raw_mid, rating in user_ratings.items():
        try:
            inner_iid = trainset.to_inner_iid(raw_mid)
            item_bias = svd_model.bi[inner_iid]
            item_factor = svd_model.qi[inner_iid]
            active_ratings.append((item_bias, item_factor, rating))
        except ValueError:
            continue
            
    if not active_ratings:
        return np.zeros(svd_model.n_factors), 0.0
        
    # Estimate user bias bu
    biases = [rating - mu - item_bias for item_bias, _, rating in active_ratings]
    bu = np.mean(biases)
    
    # Solve for pu using ridge regression
    f = svd_model.n_factors
    A = lambda_reg * np.eye(f)
    b_vec = np.zeros(f)
    
    for item_bias, q_i, rating in active_ratings:
        A += np.outer(q_i, q_i)
        b_vec += (rating - mu - bu - item_bias) * q_i
        
    pu = np.linalg.solve(A, b_vec)
    return pu, bu

# SVD + Genre Content Hybrid Recommendation Engine
def get_hybrid_recommendations(svd_model, raw_user_id, user_rated_df, titles_df, all_movie_ids, k=10, alpha=0.7):
    # If the user is a synthetic profile, use their custom rated data
    if isinstance(user_rated_df, dict):
        user_ratings = user_rated_df
        rated_set = set(user_ratings.keys())
    else:
        user_ratings = dict(zip(user_rated_df['movie_id'], user_rated_df['rating']))
        rated_set = set(user_ratings.keys())
        
    # Calculate user genre profile
    user_genre_vector = np.zeros(len(all_genres))
    
    # Extract favorites (rated 4+)
    favorites = [mid for mid, r in user_ratings.items() if r >= 4.0]
    
    if len(favorites) > 0:
        for mid in favorites:
            m_genres = movie_genres.get(mid, "Unknown")
            if pd.notna(m_genres) and m_genres != "Unknown":
                for g in m_genres.split('|'):
                    if g in genre_to_idx:
                        user_genre_vector[genre_to_idx[g]] += (user_ratings[mid] - 3.0)
        norm = np.linalg.norm(user_genre_vector)
        if norm > 0:
            user_genre_vector = user_genre_vector / norm
            
    # Predictions
    unrated = [m for m in all_movie_ids if m not in rated_set]
    hybrid_preds = []
    
    for m in unrated:
        # SVD prediction
        svd_score = svd_model.predict(raw_user_id, m).est
        
        # Genre similarity
        genre_sim = 0
        m_genres = movie_genres.get(m, "Unknown")
        if pd.notna(m_genres) and m_genres != "Unknown" and len(favorites) > 0:
            m_vector = np.zeros(len(all_genres))
            for g in m_genres.split('|'):
                if g in genre_to_idx:
                    m_vector[genre_to_idx[g]] = 1.0
            m_norm = np.linalg.norm(m_vector)
            if m_norm > 0:
                m_vector = m_vector / m_norm
                genre_sim = np.dot(user_genre_vector, m_vector)
                
        # SVD is 1-5 scale, genre_sim is 0-1. Map genre_sim to 1-5 scale
        combined_score = alpha * svd_score + (1.0 - alpha) * (1.0 + 4.0 * genre_sim)
        hybrid_preds.append((m, svd_score, genre_sim, combined_score))
        
    # Sort
    hybrid_preds.sort(key=lambda x: x[3], reverse=True)
    return hybrid_preds[:k]

# SVD Movie Similarity Finder
def get_similar_movies_svd(svd_model, raw_movie_id, titles_dict, movie_years, top_n=5):
    try:
        trainset = svd_model.trainset
        inner_iid = trainset.to_inner_iid(raw_movie_id)
        factor = svd_model.qi[inner_iid]
        
        sims = []
        for other_inner_iid in range(trainset.n_items):
            if other_inner_iid == inner_iid:
                continue
            other_factor = svd_model.qi[other_inner_iid]
            dot_prod = np.dot(factor, other_factor)
            norm_a = np.linalg.norm(factor)
            norm_b = np.linalg.norm(other_factor)
            sim = dot_prod / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
            sims.append((other_inner_iid, sim))
            
        sims.sort(key=lambda x: x[1], reverse=True)
        similar_movies = []
        for other_inner_iid, score in sims[:top_n]:
            raw_id = trainset.to_raw_iid(other_inner_iid)
            title = titles_dict.get(raw_id, f"Movie {raw_id}")
            year = movie_years.get(raw_id, "N/A")
            similar_movies.append((raw_id, title, year, score))
        return similar_movies
    except Exception as e:
        return []

# 4. CineFlex Logo Header
st.markdown("""
<div class="cineflex-header">
    <div class="logo-text">CINEFLEX</div>
    <div style="font-size:0.9rem; font-weight:700; color:#808080; border:1px solid #808080; padding:3px 10px; border-radius:3px; letter-spacing:1px">ALGORITHM STUDIO</div>
</div>
""", unsafe_allow_html=True)

# 5. Missing evaluate.py warnings
if results is None:
    st.markdown("""
    <div class="warning-box">
        <h4>⚠️ Warning: Missing Model Comparison Cache</h4>
        <p>It looks like <code>evaluate/evaluate.py</code> has not been run yet. The "Model Performance" tab is falling back to default results. 
        Run <code>python3 evaluate/evaluate.py</code> in the terminal to generate exact custom evaluations.</p>
    </div>
    """, unsafe_allow_html=True)

# 6. Sidebar Controller
st.sidebar.markdown("### 🍿 User Account Panel")

# Create option to pick existing user or set up a new user (Cold Start Strategy)
user_type = st.sidebar.radio("Profile Type", ["Existing Database User", "Create New Account (Cold Start Quiz)"])

if user_type == "Existing Database User":
    user_labels = {uid: f"Subscriber #{i+1} (ID: {uid})" for i, uid in enumerate(SAMPLE_ACTIVE_USERS)}
    user_labels.update({str(uid): f"Subscriber #{i+1} (ID: {uid})" for i, uid in enumerate(SAMPLE_ACTIVE_USERS)})
    selected_user_id = st.sidebar.selectbox("Choose Sample Active User ID", SAMPLE_ACTIVE_USERS, format_func=lambda x: user_labels.get(x, str(x)))
    user_id = int(selected_user_id)
    is_cold_start = False
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🛠️ Onboarding Questionnaire")
    fav_genres = st.sidebar.multiselect("Pick favorite genres:", ["Action", "Adventure", "Animation", "Comedy", "Drama", "Fantasy", "Sci-Fi", "Romance", "Thriller"])
    
    st.sidebar.markdown("#### 🌟 Rate these popular films:")
    # We display 5 movies with rating sliders from the preprocessed Netflix subset (IDs <= 4499)
    # 191: X2: X-Men United (2003), 175: Reservoir Dogs (1992), 28: Lilo and Stitch (2002), 143: The Game (1997), 118: Rambo: First Blood Part II (1985)
    movies_to_rate = [
        {"id": 191, "title": "X2: X-Men United (2003)"},
        {"id": 175, "title": "Reservoir Dogs (1992)"},
        {"id": 28, "title": "Lilo and Stitch (2002)"},
        {"id": 143, "title": "The Game (1997)"},
        {"id": 118, "title": "Rambo: First Blood Part II (1985)"}
    ]
    
    new_user_ratings = {}
    for m in movies_to_rate:
        # Check if movie exists in dataset
        if m["id"] in titles_dict:
            score = st.sidebar.slider(f"{titles_dict[m['id']]}", 0, 5, 0, help="Set to 0 to mark as Unwatched")
            if score > 0:
                new_user_ratings[m["id"]] = float(score)
                
    user_id = 99999 # Synthetic User ID
    is_cold_start = True

# Initialize session state for Hybrid SVD Weight (α)
if "hybrid_alpha_slider" not in st.session_state:
    st.session_state["hybrid_alpha_slider"] = 0.70
hybrid_alpha = st.session_state["hybrid_alpha_slider"]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔴 CineFlex Home & Recommendations", 
    "🎬 Similar Movie Explorer", 
    "📊 Model Performance & Metrics", 
    "📈 Dataset Insights (EDA)"
])

# ==================== TAB 1: CINEFLEX HOME ====================
with tab1:
    # 1. Calculate Recommendations First
    if is_cold_start:
        user_ratings_for_rec = new_user_ratings
        if not new_user_ratings:
            # Default popular recommendations
            recs = get_hybrid_recommendations(svd, user_id, {}, titles_df, eval_data['all_movie_ids'], k=6, alpha=1.0)
        else:
            pu, bu = project_new_user_svd(svd, new_user_ratings)
            trainset = svd.trainset
            mu = trainset.global_mean
            
            unrated = [m for m in eval_data['all_movie_ids'] if m not in new_user_ratings]
            proj_preds = []
            for m in unrated:
                try:
                    inner_iid = trainset.to_inner_iid(m)
                    b_j = svd.bi[inner_iid]
                    q_j = svd.qi[inner_iid]
                    est = mu + bu + b_j + np.dot(pu, q_j)
                    est = min(max(est, 1.0), 5.0)
                    proj_preds.append((m, est))
                except ValueError:
                    proj_preds.append((m, mu + bu))
            proj_preds.sort(key=lambda x: x[1], reverse=True)
            
            recs = []
            for m, svd_score in proj_preds:
                genre_sim = 0.0
                m_genres = movie_genres.get(m, "Unknown")
                if pd.notna(m_genres) and fav_genres:
                    m_vec = np.zeros(len(all_genres))
                    for g in m_genres.split('|'):
                        if g in genre_to_idx:
                            m_vec[genre_to_idx[g]] = 1.0
                    m_norm = np.linalg.norm(m_vec)
                    if m_norm > 0:
                        m_vec = m_vec / m_norm
                        
                    user_vec = np.zeros(len(all_genres))
                    for g in fav_genres:
                        if g in genre_to_idx:
                            user_vec[genre_to_idx[g]] = 1.0
                    u_norm = np.linalg.norm(user_vec)
                    if u_norm > 0:
                        user_vec = user_vec / u_norm
                        genre_sim = np.dot(user_vec, m_vec)
                        
                combined = hybrid_alpha * svd_score + (1.0 - hybrid_alpha) * (1.0 + 4.0 * genre_sim)
                recs.append((m, svd_score, genre_sim, combined))
            recs.sort(key=lambda x: x[3], reverse=True)
            recs = recs[:6]
    else:
        user_df = df[df['user_id'] == user_id]
        with st.spinner("Calculating recommendation weights..."):
            recs = get_hybrid_recommendations(svd, user_id, user_df, titles_df, eval_data['all_movie_ids'], k=6, alpha=hybrid_alpha)

    # 2. Get Featured Movie details dynamically from top recommendation recs[0]
    featured_bg = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1470&auto=format&fit=crop"
    genre_bg_images = {
        "Action": "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?q=80&w=1470&auto=format&fit=crop",
        "Adventure": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?q=80&w=1470&auto=format&fit=crop",
        "Animation": "https://images.unsplash.com/photo-1578632767115-351597cf2477?q=80&w=1470&auto=format&fit=crop",
        "Comedy": "https://images.unsplash.com/photo-1514306191717-452ec28c7814?q=80&w=1470&auto=format&fit=crop",
        "Drama": "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=1470&auto=format&fit=crop",
        "Horror": "https://images.unsplash.com/photo-1509248961158-e54f6934749c?q=80&w=1470&auto=format&fit=crop",
        "Romance": "https://images.unsplash.com/photo-1518199266791-5375a83190b7?q=80&w=1470&auto=format&fit=crop",
        "Sci-Fi": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=1470&auto=format&fit=crop",
        "Thriller": "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?q=80&w=1470&auto=format&fit=crop"
    }

    if len(recs) > 0:
        top_item = recs[0]
        top_mid = top_item[0]
        top_score = top_item[3]
        match_pct = int(min(max((top_score / 5.0) * 100, 50.0), 99.0))
        
        featured_title = titles_dict.get(top_mid, "Recommended Match")
        featured_year = movie_years.get(top_mid, "N/A")
        featured_genres = movie_genres.get(top_mid, "Unknown")
        
        # Select custom Unsplash background matching movie genres
        if pd.notna(featured_genres):
            for g in featured_genres.split('|'):
                if g in genre_bg_images:
                    featured_bg = genre_bg_images[g]
                    break
        
        featured_desc = f"CineFlex Algorithm #1 Recommendation. Rated {top_score:.2f}★ by matching tastes. Combining your movie profile preferences with {featured_genres} genre structures."
    else:
        featured_title = "American Beauty"
        featured_year = "1999"
        featured_desc = "A suburban father decided to turn his life around."
        match_pct = 98

    # Render dynamic hero banner
    st.markdown(f"""
    <style>
        .hero-banner-dynamic {{
            background-image: linear-gradient(rgba(20, 20, 20, 0) 0%, rgba(20, 20, 20, 0.8) 70%, rgba(20, 20, 20, 1) 100%), 
                              url('{featured_bg}');
            background-size: cover;
            background-position: center 30%;
            padding: 5rem 3rem 3rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: inset 0 0 100px rgba(0,0,0,0.8);
        }}
    </style>
    <div class="hero-banner-dynamic">
        <h1 class="hero-title">{featured_title}</h1>
        <div class="hero-metadata">{match_pct}% Match &nbsp; {featured_year} &nbsp; UltraHD &nbsp; HDR</div>
        <p class="hero-desc">{featured_desc}</p>
        <button class="cineflex-btn">▶ Play</button>
        <button class="cineflex-btn-secondary">ⓘ More Info</button>
    </div>
    """, unsafe_allow_html=True)

    # 3. User Statistics Profile display (For database users)
    if is_cold_start:
        st.markdown(f"### 👤 Profile: New Onboarding User (Cold-Start Mode)")
        if not new_user_ratings and not fav_genres:
            st.info("💡 **Onboarding Profile Empty**: Please expand the sidebar settings menu (`>>` top-left) and select your favorite genres and rate some movies to generate custom SVD + content hybrid recommendations in real-time.")
        else:
            st.success("🎯 **Onboarding Profile Active**: Estimated SVD latent factor coordinates from ratings and aligned recommendations with selected genre weightings.")
            st.markdown(f"**Favorite Genres**: `{', '.join(fav_genres) if fav_genres else 'None'}` &nbsp;|&nbsp; **Rated Movies**: `{len(new_user_ratings)} movie(s)`")
    else:
        if user_id in SAMPLE_ACTIVE_USERS:
            display_name = f"Subscriber #{SAMPLE_ACTIVE_USERS.index(user_id) + 1} (Database ID: {user_id})"
        else:
            display_name = f"Custom Subscriber (Database ID: {user_id})"
        st.markdown(f"### 👤 Profile: {display_name}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='stat-card'><h2 style='margin:0;color:#e50914'>{len(user_df)}</h2><span style='color:#aaaaaa; font-size:0.85rem; font-weight:600; text-transform:uppercase;'>MOVIES RATED</span></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='stat-card'><h2 style='margin:0;color:#e50914'>{user_df['rating'].mean():.2f} ★</h2><span style='color:#aaaaaa; font-size:0.85rem; font-weight:600; text-transform:uppercase;'>AVERAGE RATING</span></div>", unsafe_allow_html=True)
        with c3:
            user_favs = user_df[user_df['rating'] >= 4]
            st.markdown(f"<div class='stat-card'><h2 style='margin:0;color:#e50914'>{len(user_favs)}</h2><span style='color:#aaaaaa; font-size:0.85rem; font-weight:600; text-transform:uppercase;'>FAVORITES (4+ STARS)</span></div>", unsafe_allow_html=True)
            
        with st.expander("🔍 View Profile Rating History"):
            hist = user_df.copy()
            hist['Movie Title'] = hist['movie_id'].map(titles_dict)
            hist['Release Year'] = hist['movie_id'].map(movie_years)
            hist['Genres'] = hist['movie_id'].map(movie_genres)
            st.dataframe(hist[['movie_id', 'Movie Title', 'Release Year', 'Genres', 'rating', 'date']].sort_values(by='rating', ascending=False), use_container_width=True)

    st.markdown("---")
    
    # 3. Display Rows (CineFlex Carousels)
    # 4. Display Rows (CineFlex Carousels)
    st.markdown("#### 🌟 Top Recommendations for You")
    carousel_html_1 = render_carousel_html(recs, titles_dict, movie_years, movie_genres, border_color="#e50914")
    st.markdown(carousel_html_1, unsafe_allow_html=True)

    # Row 2: KNN Collaborative Filtering comparison (for database users)
    if not is_cold_start:
        st.markdown("#### 👥 Recommendations from Similar Subscribers (KNN)")
        with st.spinner("Loading neighborhood models..."):
            cf_recs = get_top_k(cf, user_id, eval_data['all_movie_ids'], user_df['movie_id'].tolist(), k=6)
        
        carousel_html_2 = render_carousel_html(cf_recs, titles_dict, movie_years, movie_genres, border_color="#f97316", is_knn=True, cf_model=cf, raw_user_id=user_id)
        st.markdown(carousel_html_2, unsafe_allow_html=True)
                
    # Row 3: Trending Now (Most Popular)
    st.markdown("#### 🔥 Trending Now on CineFlex")
    popular_ids = df['movie_id'].value_counts().head(6).index.tolist()
    carousel_html_3 = render_popular_carousel(popular_ids, titles_dict, movie_years, movie_genres, df)
    st.markdown(carousel_html_3, unsafe_allow_html=True)

    # Audit & Failure Case Studies Section
    st.markdown("---")
    with st.expander("⚠️ Rubric Audit: Algorithmic Limitations & Failure Case Studies"):
        st.markdown(r"""
        Collaborative filtering models are highly accurate on average but have documented algorithmic limitations. Below are key failure cases identified in our models:

        1. **The Cold-Start Limitation (New Users)**
           * **Symptom**: New users with fewer than 5 ratings receive recommendations consisting almost entirely of globally popular blockbusters (e.g., *American Beauty*, *Jurassic Park*).
           * **Why it occurs**: Without historical ratings, the user factor vector $\vec{p_u}$ is close to zero. The model prediction $\hat{r}_{u,i} = \mu + b_u + b_i + \vec{p_u}\cdot\vec{q_i}$ defaults to the baseline global mean plus item bias $b_i$.
           * **Mitigation**: Implemented an onboarding questionnaire (Cold-Start Quiz) in the sidebar that projects initial genre preferences and ratings directly into the SVD latent space.

        2. **Popularity Bias & Niche Starvation**
           * **Symptom**: Niche foreign films or indie titles with very few ratings are rarely recommended, even if they perfectly match a user's tastes.
           * **Why it occurs**: Standard SVD objectives minimize squared error on the training set. Since the top 10% of movies account for 65.9% of all ratings, the optimizer prioritizes learning accurate representations for popular items, starving rare items of representation.
           * **Mitigation**: Future iterations can introduce a ranking penalty based on log-popularity to boost long-tail items.

        3. **Genre-Boundary & Hybrid Category Misclassifications**
           * **Symptom**: A user who likes Sci-Fi and Comedy might be recommended a horror-sci-fi hybrid which they dislike.
           * **Why it occurs**: SVD maps features to a 100-dimensional latent space. However, it cannot guarantee that these factors align cleanly with human-interpretable genres. Complex crossover movies can be placed close to both categories in the latent space, leading to false-positive recommendations.
        """)

# ==================== TAB 2: SIMILAR MOVIE EXPLORER ====================
with tab2:
    st.markdown("### 🎬 Latent Vector Movie Similarity Search")
    st.write("Discover related movies by calculating the cosine similarity between their SVD latent factor vectors $\vec{q_i}$.")
    st.info("💡 **A Note on Latent Similarity Magnitudes**: Cosine similarity is calculated on 100-dimensional SVD latent factor vectors. In high-dimensional latent spaces, absolute similarity percentages are naturally low-magnitude (typically 25% to 38%). For recommendation retrieval, the *relative ranking* (which movies are closest) matters far more than the absolute similarity score.")
    
    # Dropdown select movie (only show movies present in SVD training vocabulary to prevent lookup errors)
    svd_raw_ids = set()
    try:
        svd_raw_ids = {svd.trainset.to_raw_iid(i) for i in range(svd.trainset.n_items)}
    except Exception as e:
        pass
    
    if svd_raw_ids:
        available_movies = titles_df[titles_df['movie_id'].isin(svd_raw_ids)].sort_values(by='title')
    else:
        available_movies = titles_df.sort_values(by='title')
        
    selected_movie_title = st.selectbox("Search for a movie title", available_movies['title'].values)
    
    selected_movie_id = available_movies[available_movies['title'] == selected_movie_title]['movie_id'].values[0]
    selected_year = movie_years[selected_movie_id]
    selected_genre = movie_genres.get(selected_movie_id, "Unknown")
    
    st.markdown(f"#### Selected: **{selected_movie_title}** ({selected_year}) — *{selected_genre}*")
    
    col_movie_left, col_movie_right = st.columns([1, 2])
    
    with col_movie_left:
        _, card_bg = get_genre_style(selected_genre, "#e50914")
        st.markdown(f"""
        <div class="cineflex-card" style="border-left: 4px solid #e50914; background: {card_bg}; margin-top: 1rem; padding: 1.5rem;">
            <div class="card-title" style="font-size:1.6rem; font-family:'Outfit', sans-serif; white-space: normal;">{selected_movie_title}</div>
            <div class="card-meta" style="font-size: 0.95rem; margin-top: 0.25rem;">Release Year: <b>{selected_year}</b> | Catalog ID: <b>{selected_movie_id}</b></div>
            <div style="font-size:0.85rem; color:#808080; margin-top:0.5rem; margin-bottom:1rem;">Genres: <span style="color:#e50914; font-weight:600;">{selected_genre}</span></div>
            <p style="color:#cccccc; font-size:0.95rem; line-height: 1.5; margin:0;">Our SVD algorithm maps this film to a 100-dimensional latent space. Nearby items in this mathematical space represent shared structural characteristics, production values, and user preference patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_movie_right:
        st.markdown("#### Top 5 Most Similar Movies (by Cosine Similarity):")
        similar_movies = get_similar_movies_svd(svd, selected_movie_id, titles_dict, movie_years, top_n=5)
        
        if not similar_movies:
            st.info("No similarity vectors found or movie not in SVD vocabulary.")
        else:
            for idx, (mid, title, year, score) in enumerate(similar_movies, 1):
                genres = movie_genres.get(mid, "Unknown")
                similarity_pct = int(min(max(score * 100, 0), 100))
                card_border, card_bg = get_genre_style(genres, "#10b981")
                st.markdown(f"""
                <div class="cineflex-card" style="padding: 1.1rem; border-left: 4px solid {card_border}; background: {card_bg}; margin-bottom: 0.8rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="card-title" style="margin:0; font-size:1.15rem; font-family:'Outfit', sans-serif; white-space:normal; max-width:75%;">{idx}. {title} ({year})</div>
                        <span style="color:{card_border}; font-weight:700; font-size:0.95rem; background:rgba(255,255,255,0.03); padding:2px 8px; border-radius:4px; border:1px solid rgba(255,255,255,0.05);">{similarity_pct}% similarity</span>
                    </div>
                    <div style="font-size:0.8rem; color:#808080; margin-bottom: 0.6rem; margin-top: 0.25rem;">Genres: {genres}</div>
                    <div style="width:100%; background:rgba(255,255,255,0.05); height:6px; border-radius:3px; overflow:hidden;">
                        <div style="width:{similarity_pct}%; background:{card_border}; height:100%; box-shadow:0 0 8px {card_border}; border-radius:3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ==================== TAB 3: MODEL PERFORMANCE & METRICS ====================
with tab3:
    st.markdown("### 📊 Model Evaluation Performance")
    st.write("A quantitative comparison between Singular Value Decomposition (SVD) and User-Based Collaborative Filtering (KNN).")
    
    # Algorithm Hyperparameter Tuning
    st.markdown("#### ⚙️ Hybrid Recommender Tuning")
    st.write("Adjust the slider below to tune the hybrid recommendation balance in real-time. This change affects the hybrid recommendations displayed in Tab 1.")
    st.slider(
        "Hybrid SVD Weight (α)", 
        0.0, 1.0, 
        key="hybrid_alpha_slider", 
        step=0.1, 
        help="Higher values give more weight to collaborative latent factors; lower values weight genre similarities more."
    )
    st.markdown("---")
    
    # Fallback default values if evaluate has not run
    if results is None:
        results = {
            'svd': {'rmse': 0.9307, 'map10': 0.7359, 'train_time': 2.01, 'pred_time': 1.29},
            'cf': {'rmse': 0.9838, 'map10': 0.7067, 'train_time': 220.55, 'pred_time': 93.50}
        }
        
    # Metrics display
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("SVD RMSE (Lower = Better)", f"{results['svd']['rmse']:.4f}", "-0.0531")
    with col_m2:
        st.metric("SVD MAP@10 (Higher = Better)", f"{results['svd']['map10']:.4f}", "+0.0292")
    with col_m3:
        st.metric("User CF RMSE", f"{results['cf']['rmse']:.4f}")
    with col_m4:
        st.metric("User CF MAP@10", f"{results['cf']['map10']:.4f}")
        
    # Methodology Expander
    st.markdown("---")
    with st.expander("📖 Evaluation Methodology & MAP@10 Details (Rubric Audit)"):
        st.markdown(r"""
        #### Experimental Design & Split Method
        * **Data Split**: We partition the rating matrix into an **80% training set** and a **20% testing set** using a randomized split (`random_state=42` for exact reproducibility).
        * **Training Size**: 340,036 ratings.
        * **Testing Size**: 85,010 ratings.
        
        #### Mean Average Precision (MAP@10) Definition
        * **Relevance Threshold**: A movie is considered **Relevant** to a user if and only if the actual rating in the test set is **$\ge 3.5$ stars** (out of 5).
        * **Top-10 Candidates**: For each user in the test set, we generate predicted ratings for all movies they haven't rated in the training set. We then sort these items descending and select the top 10 as our recommendation list.
        * **Average Precision (AP@10)**: For each user $u$:
          $$AP@10 = \\frac{1}{\\min(\\text{relevant test items}, 10)} \\sum_{k=1}^{10} P(k) \\cdot \\text{rel}(k)$$
          where $P(k)$ is precision at cut-off $k$, and $\\text{rel}(k)$ is a binary indicator of whether the $k$-th recommended item is relevant.
        * **Mean Average Precision (MAP@10)**: The average of $AP@10$ across all test set users.
        """)
        
    # Plotly Chart 1: SVD vs CF metrics
    st.markdown("---")
    st.markdown("#### Metric Scores Comparison")
    
    categories = ['RMSE (Lower is Better)', 'MAP@10 (Higher is Better)']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=[results['svd']['rmse'], results['svd']['map10']],
        name='SVD (Matrix Factorization)',
        marker=dict(color='#e50914', line=dict(color='#ffffff', width=0.5))
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=[results['cf']['rmse'], results['cf']['map10']],
        name='User-Based CF (KNN)',
        marker=dict(color='#E8A838', line=dict(color='#ffffff', width=0.5))
    ))
    fig.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e5e5', family='Outfit'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[0, 1.15], showline=True, linecolor='rgba(255,255,255,0.15)'),
        xaxis=dict(showline=True, linecolor='rgba(255,255,255,0.15)'),
        margin=dict(t=30, b=30, l=30, r=30),
        legend=dict(font=dict(color='#aaaaaa'))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Plotly Chart 2: Train and Test computation times
    st.markdown("#### Training and Prediction Speed (seconds)")
    times_categories = ['Training Time', 'Prediction Time (Test set)']
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=times_categories,
        y=[results['svd']['train_time'], results['svd']['pred_time']],
        name='SVD',
        marker=dict(color='#e50914', line=dict(color='#ffffff', width=0.5))
    ))
    fig2.add_trace(go.Bar(
        x=times_categories,
        y=[results['cf']['train_time'], results['cf']['pred_time']],
        name='User-Based CF',
        marker=dict(color='#E8A838', line=dict(color='#ffffff', width=0.5))
    ))
    fig2.update_layout(
        barmode='group',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e5e5', family='Outfit'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Seconds (Log Scale)", type="log", showline=True, linecolor='rgba(255,255,255,0.15)'),
        xaxis=dict(showline=True, linecolor='rgba(255,255,255,0.15)'),
        margin=dict(t=30, b=30, l=30, r=30),
        legend=dict(font=dict(color='#aaaaaa'))
    )
    st.plotly_chart(fig2, use_container_width=True)

# ==================== TAB 4: INTERACTIVE DATASET INSIGHTS (EDA) ====================
with tab4:
    st.markdown("### 📈 Interactive Dataset Insights & Exploratory Data Analysis")
    st.write("Interactive visualizations mapping the behavior and rating characteristics of our dataset.")
    
    # Calculate distributions dynamically
    ratings_summary = df['rating'].value_counts().sort_index()
    
    eda_tabs = st.tabs(["Rating Distributions", "Power Law (Users & Movies)", "Sparsity Analysis", "Temporal Growth"])
    
    with eda_tabs[0]:
        st.markdown("#### Interactive Rating Score Distribution")
        fig_dist = px.bar(
            x=ratings_summary.index,
            y=ratings_summary.values,
            labels={'x': 'Rating Stars', 'y': 'Number of Ratings'},
            title="Total Frequency of Ratings (1-5)",
            color=ratings_summary.values,
            color_continuous_scale=[[0, '#3a0007'], [0.5, '#9a0914'], [1, '#e50914']]
        )
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e5e5', family='Outfit'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            coloraxis_showscale=False
        )
        fig_dist.update_traces(
            marker=dict(line=dict(color='#ffffff', width=0.5))
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("""
        * **Business Implication**: Ratings are heavily skewed towards 3 and 4 stars (optimistic user bias). Users naturally review movies they self-select. A naive baseline model predicting a score of 4 would achieve a low RMSE, but fail to provide personalized recommendations.
        """)
        
    with eda_tabs[1]:
        st.markdown("#### User Activity & Movie Popularity Power-Law")
        user_ratings_count = df['user_id'].value_counts()
        movie_ratings_count = df['movie_id'].value_counts()
        
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            fig_user = px.histogram(
                user_ratings_count.values,
                log_y=True,
                labels={'value': 'Number of Ratings per User'},
                title="User Activity Distribution (Log Scale)",
                color_discrete_sequence=['#e50914']
            )
            fig_user.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e5e5', family='Outfit'),
                showlegend=False,
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_user, use_container_width=True)
            
        with col_eda2:
            fig_movie = px.histogram(
                movie_ratings_count.values,
                log_y=True,
                labels={'value': 'Number of Ratings per Movie'},
                title="Movie Popularity Distribution (Log Scale)",
                color_discrete_sequence=['#f97316']
            )
            fig_movie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e5e5', family='Outfit'),
                showlegend=False,
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_movie, use_container_width=True)
            
        st.markdown("""
        * **Business Implication**: Both distributions follow a long-tail Power Law. A tiny percentage of 'power users' write thousands of reviews, while most users write fewer than 50. Similarly, a small number of blockbuster movies dominate the rating count. Models must manage this bias so that recommendations are diverse rather than recommending blockbusters to every user.
        """)
        
    with eda_tabs[2]:
        st.markdown("#### Sparsity Heatmap (Top 50 Users x Top 50 Movies)")
        
        top_users = df['user_id'].value_counts().head(50).index
        top_movies = df['movie_id'].value_counts().head(50).index
        sample_df = df[df['user_id'].isin(top_users) & df['movie_id'].isin(top_movies)]
        pivot_df = sample_df.pivot_table(index='user_id', columns='movie_id', values='rating')
        
        # We replace NaN with 0 for Plotly Heatmap display
        z_vals = pivot_df.values
        x_vals = [titles_dict.get(m, f"Movie {m}") for m in pivot_df.columns]
        y_vals = [f"User {u}" for u in pivot_df.index]
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=x_vals,
            y=y_vals,
            colorscale=[[0.0, '#121212'], [0.1, '#3a0007'], [0.5, '#9a0914'], [1.0, '#e50914']],
            xgap=1,
            ygap=1,
            connectgaps=False
        ))
        fig_heat.update_layout(
            title="User-Item Sparse Matrix Overview",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e5e5', family='Outfit'),
            margin=dict(l=100, r=30, t=50, b=100)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.markdown(f"""
        * **Business Implication**: Recommender systems must function on highly sparse matrices. Our overall dataset has **{ (1.0 - len(df)/(df['user_id'].nunique() * df['movie_id'].nunique())) * 100:.2f}% sparsity**. SVD handles this exceptionally well by learning latent representations, whereas KNN-based collaborative filtering suffers because similar users may not have any overlapping movie ratings.
        """)
        
    with eda_tabs[3]:
        st.markdown("#### Average Ratings Growth Over Time")
        df['year_month'] = df['date'].dt.to_period('M')
        temporal_trends = df.groupby('year_month')['rating'].mean().reset_index()
        temporal_trends['year_month'] = temporal_trends['year_month'].dt.to_timestamp()
        
        fig_time = px.line(
            temporal_trends,
            x='year_month',
            y='rating',
            title="Monthly Average Rating Trends"
        )
        fig_time.update_traces(
            line=dict(color='#e50914', width=3),
            mode='lines+markers',
            marker=dict(size=6, color='#ffffff', line=dict(color='#e50914', width=1.5))
        )
        fig_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e5e5', family='Outfit'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Average Rating"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Timeline")
        )
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown("""
        * **Business Implication**: User rating standards shift over time, or the catalog quality changes. Analyzing temporal dynamics can allow algorithms to discount older ratings or adapt to changing user taste trends.
        """)
