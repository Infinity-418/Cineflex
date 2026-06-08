import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_netflix_file(filepath):
    print(f"Parsing raw ratings from {filepath}...")
    rows, movie_id = [], None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                movie_id = int(line[:-1])
            else:
                uid, rating, date = line.split(',')
                rows.append((int(uid), movie_id, int(rating), date))
    df = pd.DataFrame(rows, columns=['user_id', 'movie_id', 'rating', 'date'])
    # Convert types for optimization
    df['user_id'] = df['user_id'].astype(np.int32)
    df['movie_id'] = df['movie_id'].astype(np.int32)
    df['rating'] = df['rating'].astype(np.int8)
    df['date'] = pd.to_datetime(df['date'])
    return df

def run_eda():
    data_dir = os.path.join(os.getcwd(), 'data')
    plots_dir = os.path.join(os.getcwd(), 'eda', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    ratings_file = os.path.join(data_dir, 'combined_data_1.txt')
    titles_file = os.path.join(data_dir, 'movie_titles.csv')
    
    if not os.path.exists(ratings_file) or not os.path.exists(titles_file):
        print("Error: Dataset files not found. Run download_and_prepare.py first.")
        return

    # Load data
    df = parse_netflix_file(ratings_file)
    titles = pd.read_csv(titles_file, header=None, names=['movie_id', 'year', 'title', 'genres'], encoding='latin-1', on_bad_lines='skip')
    
    n_users = df['user_id'].nunique()
    n_movies = df['movie_id'].nunique()
    n_ratings = len(df)
    
    print("\n--- DATASET SUMMARY ---")
    print(f"Total Ratings: {n_ratings:,}")
    print(f"Unique Users: {n_users:,}")
    print(f"Unique Movies: {n_movies:,}")
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.rcParams['font.size'] = 12

    # 1. Rating Distribution
    print("Generating Rating Distribution plot...")
    plt.figure()
    rating_counts = df['rating'].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, hue=rating_counts.index, palette="Reds_d", legend=False)
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating (Stars)")
    plt.ylabel("Number of Ratings")
    for i, val in enumerate(rating_counts.values):
        plt.text(i, val + n_ratings*0.01, f"{val/n_ratings:.1%}", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rating_distribution.png'), dpi=150)
    plt.close()
    
    # Business/Technical Implication
    mean_rating = df['rating'].mean()
    print(f"Mean Rating: {mean_rating:.2f}")
    print("Implication: Ratings skew towards 3-4. A baseline predicting 4 will get low RMSE, but fail to provide useful recommendations.")

    # 2. User Activity (Power Law)
    print("Generating User Activity plot...")
    plt.figure()
    user_ratings = df['user_id'].value_counts()
    sns.histplot(user_ratings.values, log_scale=True, kde=True, color="crimson")
    plt.title("User Activity Distribution (Log Scale)")
    plt.xlabel("Number of Ratings per User")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'user_activity.png'), dpi=150)
    plt.close()
    
    # 3. Movie Popularity
    print("Generating Movie Popularity plot...")
    plt.figure()
    movie_ratings = df['movie_id'].value_counts()
    sns.histplot(movie_ratings.values, log_scale=True, kde=True, color="firebrick")
    plt.title("Movie Popularity Distribution (Log Scale)")
    plt.xlabel("Number of Ratings per Movie")
    plt.ylabel("Number of Movies")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'movie_popularity.png'), dpi=150)
    plt.close()

    # Share of top 100 movies
    top_100_share = movie_ratings.head(100).sum() / n_ratings
    print(f"Top 100 movies account for {top_100_share:.1%} of all ratings.")
    
    # Print most rated movie titles
    print("\nTop 5 Most Rated Movies:")
    top_5_ids = movie_ratings.head(5).index
    for i, mid in enumerate(top_5_ids, 1):
        m_title = titles[titles['movie_id'] == mid]['title'].values[0]
        m_year = titles[titles['movie_id'] == mid]['year'].values[0]
        print(f"{i}. {m_title} ({m_year}) - {movie_ratings[mid]:,} ratings")

    # 4. Sparsity Analysis
    sparsity = 1.0 - (n_ratings / (n_users * n_movies))
    print(f"Sparsity: {sparsity * 100:.4f}%")
    
    print("Generating Sparsity Heatmap...")
    # Get a sample of top 50 users and top 50 movies to visualize
    top_users = df['user_id'].value_counts().head(50).index
    top_movies = df['movie_id'].value_counts().head(50).index
    sample_df = df[df['user_id'].isin(top_users) & df['movie_id'].isin(top_movies)]
    pivot_df = sample_df.pivot_table(index='user_id', columns='movie_id', values='rating')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_df, cmap="YlOrRd", cbar=True, annot=False, mask=pivot_df.isnull(), 
                linewidths=.5, linecolor='gray')
    plt.title("User-Item Interaction Heatmap (Sample of 50x50 Active Users/Movies)")
    plt.xlabel("Movie ID")
    plt.ylabel("User ID")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'sparsity_heatmap.png'), dpi=150)
    plt.close()

    # 5. Temporal Trends
    print("Generating Temporal Trends plot...")
    plt.figure()
    df['year_month'] = df['date'].dt.to_period('M')
    temporal_trends = df.groupby('year_month')['rating'].mean()
    temporal_trends.index = temporal_trends.index.to_timestamp()
    plt.plot(temporal_trends.index, temporal_trends.values, marker='o', color='darkred', linewidth=2)
    plt.title("Average Rating Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Rating")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'temporal_trends.png'), dpi=150)
    plt.close()

    # 6. Cold-Start Preview
    # In Netflix dataset, all users have >= 20 ratings. Let's count how many users have fewer than 25 ratings
    fewer_than_25 = (user_ratings < 25).sum()
    print(f"Users with fewer than 25 ratings: {fewer_than_25} ({fewer_than_25/n_users:.1%})")

    # Save summary stats to a text file
    with open(os.path.join(plots_dir, 'summary_statistics.txt'), 'w') as f:
        f.write(f"Total Ratings: {n_ratings}\n")
        f.write(f"Unique Users: {n_users}\n")
        f.write(f"Unique Movies: {n_movies}\n")
        f.write(f"Mean Rating: {mean_rating:.4f}\n")
        f.write(f"Sparsity: {sparsity * 100:.4f}%\n")
        f.write(f"Top 100 Movies Rating Share: {top_100_share * 100:.2f}%\n")

    print("\nEDA completed. All plots saved to eda/plots/")

if __name__ == '__main__':
    run_eda()
