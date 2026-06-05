import os
import time
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNWithMeans
from surprise.model_selection import train_test_split
import joblib

def parse_netflix_file(filepath):
    print(f"Parsing ratings from {filepath}...")
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                movie_id = int(line[:-1])
            else:
                uid, rating, _ = line.split(',')
                rows.append((int(uid), movie_id, int(rating)))
    df = pd.DataFrame(rows, columns=['user_id', 'movie_id', 'rating'])
    # Optimize types
    df['user_id'] = df['user_id'].astype(np.int32)
    df['movie_id'] = df['movie_id'].astype(np.int32)
    df['rating'] = df['rating'].astype(np.int8)
    return df

def train_models():
    data_dir = os.path.join(os.getcwd(), 'data')
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    ratings_file = os.path.join(data_dir, 'combined_data_1.txt')
    
    if not os.path.exists(ratings_file):
        print("Error: ratings file not found. Run download_and_prepare.py first.")
        return

    # Load ratings
    df = parse_netflix_file(ratings_file)
    
    # Load into Surprise
    print("Loading data into Surprise...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)
    
    # Train-test split (80/20 random split)
    print("Splitting data into train/test sets...")
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # 1. Train SVD model
    print("\n--- Training Model 1: SVD (Matrix Factorization) ---")
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    start_time = time.time()
    svd.fit(trainset)
    svd_train_time = time.time() - start_time
    print(f"SVD training completed in {svd_train_time:.2f} seconds.")
    
    # Save SVD model
    svd_path = os.path.join(models_dir, 'svd_model.joblib')
    joblib.dump(svd, svd_path)
    print(f"SVD model saved to {svd_path}")
    
    # 2. Train KNN model (User-Based Collaborative Filtering)
    print("\n--- Training Model 2: User-Based Collaborative Filtering (KNNWithMeans) ---")
    # Using cosine similarity, user_based = True
    cf = KNNWithMeans(k=30, sim_options={'name': 'cosine', 'user_based': True}, verbose=False)
    start_time = time.time()
    cf.fit(trainset)
    cf_train_time = time.time() - start_time
    print(f"User-Based CF training completed in {cf_train_time:.2f} seconds.")
    
    # Save KNN model
    cf_path = os.path.join(models_dir, 'cf_model.joblib')
    joblib.dump(cf, cf_path)
    print(f"User-Based CF model saved to {cf_path}")
    
    # Save test set and metadata for evaluation
    eval_data = {
        'testset': testset,
        'svd_train_time': svd_train_time,
        'cf_train_time': cf_train_time,
        'all_movie_ids': df['movie_id'].unique().tolist(),
        'all_user_ids': df['user_id'].unique().tolist()
    }
    eval_data_path = os.path.join(models_dir, 'eval_data.joblib')
    joblib.dump(eval_data, eval_data_path)
    print(f"Evaluation metadata saved to {eval_data_path}")
    print("\nModel training and serialization completed!")

if __name__ == '__main__':
    train_models()
