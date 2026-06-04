import os
import urllib.request
import zipfile
import re
import pandas as pd
from datetime import datetime

def download_and_prepare():
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    zip_path = os.path.join(data_dir, 'ml-1m.zip')
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    
    # Step 1: Download the dataset
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens 1M dataset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download completed.")
    else:
        print("MovieLens 1M zip file already exists.")
        
    # Step 2: Unzip the files
    unzip_dir = os.path.join(data_dir, 'ml-1m')
    if not os.path.exists(unzip_dir):
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction completed.")
    else:
        print("Extracted folder ml-1m already exists.")

    # Step 3: Load movies.dat
    movies_path = os.path.join(unzip_dir, 'movies.dat')
    print("Parsing movie titles...")
    movies = []
    # format: MovieID::Title::Genres
    # Title usually contains (Year), e.g. "Toy Story (1995)"
    # We use latin-1 encoding because ml-1m files contain non-ascii characters
    with open(movies_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 2:
                movie_id = int(parts[0])
                title_with_year = parts[1]
                genres = parts[2] if len(parts) >= 3 else "Unknown"
                
                # Extract year
                match = re.search(r'\((\d{4})\)', title_with_year)
                if match:
                    year = match.group(1)
                    title = title_with_year.replace(f'({year})', '').strip()
                else:
                    year = '2000' # Default fallback
                    title = title_with_year.strip()
                
                movies.append((movie_id, year, title, genres))
                
    movies_df = pd.DataFrame(movies, columns=['movie_id', 'year', 'title', 'genres'])
    
    # Step 4: Load ratings.dat
    ratings_path = os.path.join(unzip_dir, 'ratings.dat')
    print("Parsing ratings...")
    ratings = []
    # format: UserID::MovieID::Rating::Timestamp
    with open(ratings_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = int(parts[2])
                timestamp = int(parts[3])
                
                # Convert timestamp to YYYY-MM-DD
                dt = datetime.fromtimestamp(timestamp)
                date_str = dt.strftime('%Y-%m-%d')
                
                ratings.append((user_id, movie_id, rating, date_str))
                
    ratings_df = pd.DataFrame(ratings, columns=['user_id', 'movie_id', 'rating', 'date'])
    print(f"Loaded {len(ratings_df)} raw ratings.")
    
    # Step 5: Filter data (users >= 20 ratings, movies >= 50 ratings)
    print("Filtering data...")
    # Count ratings per user
    user_counts = ratings_df['user_id'].value_counts()
    active_users = user_counts[user_counts >= 20].index
    
    # Count ratings per movie
    movie_counts = ratings_df['movie_id'].value_counts()
    popular_movies = movie_counts[movie_counts >= 50].index
    
    # Filter
    filtered_ratings = ratings_df[
        ratings_df['user_id'].isin(active_users) & 
        ratings_df['movie_id'].isin(popular_movies)
    ]
    print(f"Filtered to {len(filtered_ratings)} ratings (Users with >= 20, Movies with >= 50).")
    
    # Only keep movies that are in our filtered ratings list
    active_movie_ids = filtered_ratings['movie_id'].unique()
    filtered_movies = movies_df[movies_df['movie_id'].isin(active_movie_ids)]
    
    # Step 6: Write to Netflix Prize formatted files
    print("Writing Netflix-formatted files...")
    
    # 6.1 Save movie_titles.csv
    movie_titles_path = os.path.join(data_dir, 'movie_titles.csv')
    # Save as MovieID,Year,Title (comma separated)
    filtered_movies.to_csv(movie_titles_path, index=False, header=False)
    print(f"Saved movie titles to {movie_titles_path}")
    
    # 6.2 Save combined_data_1.txt
    combined_path = os.path.join(data_dir, 'combined_data_1.txt')
    
    # Sort ratings by movie_id then user_id for structure
    filtered_ratings = filtered_ratings.sort_values(by=['movie_id', 'user_id'])
    
    with open(combined_path, 'w') as f:
        # Group by movie_id
        grouped = filtered_ratings.groupby('movie_id')
        for movie_id, group in grouped:
            f.write(f"{movie_id}:\n")
            for _, row in group.iterrows():
                f.write(f"{row['user_id']},{row['rating']},{row['date']}\n")
                
    print(f"Saved Netflix-formatted ratings to {combined_path}")
    print("Data preparation complete!")

if __name__ == '__main__':
    download_and_prepare()
