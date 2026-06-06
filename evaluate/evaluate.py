import os
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from surprise import accuracy
import joblib

def map_at_k(predictions, k=10, threshold=3.5):
    """
    MAP@10 Algorithm - Step by Step
    1. Group the test set by user_id.
    2. For each user: predict ratings for all movies in their test portion.
    3. Rank predictions by estimated rating (highest first) -> Top-10 list.
    4. Mark each recommendation as Relevant (actual >= 3.5) or Not Relevant.
    5. Compute Average Precision@10 for that user.
    6. Average AP across all users -> MAP@10.
    """
    user_items = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_items[uid].append((est, true_r))
        
    aps = []
    for uid, items in user_items.items():
        # Sort by estimated rating (highest first)
        items.sort(key=lambda x: x[0], reverse=True)
        top_k = items[:k]
        
        hits, precision_sum = 0, 0
        for i, (est, true_r) in enumerate(top_k, 1):
            if true_r >= threshold:
                hits += 1
                precision_sum += hits / i
                
        if hits > 0:
            aps.append(precision_sum / min(hits, k))
        else:
            aps.append(0)
            
    return sum(aps) / len(aps) if aps else 0

def run_evaluation():
    models_dir = os.path.join(os.getcwd(), 'models')
    eval_data_path = os.path.join(models_dir, 'eval_data.joblib')
    svd_model_path = os.path.join(models_dir, 'svd_model.joblib')
    cf_model_path = os.path.join(models_dir, 'cf_model.joblib')
    
    if not os.path.exists(eval_data_path) or not os.path.exists(svd_model_path) or not os.path.exists(cf_model_path):
        print("Error: Models and eval metadata not found. Run train.py first.")
        return

    # Load metadata and models
    print("Loading models and test set...")
    eval_data = joblib.load(eval_data_path)
    testset = eval_data['testset']
    svd_train_time = eval_data['svd_train_time']
    cf_train_time = eval_data['cf_train_time']
    
    svd = joblib.load(svd_model_path)
    cf = joblib.load(cf_model_path)
    
    # Evaluate SVD
    print("\nEvaluating SVD Model...")
    start_time = time.time()
    svd_preds = svd.test(testset)
    svd_pred_time = time.time() - start_time
    rmse_svd = accuracy.rmse(svd_preds, verbose=False)
    map_svd = map_at_k(svd_preds, k=10)
    print(f"SVD RMSE: {rmse_svd:.4f}")
    print(f"SVD MAP@10: {map_svd:.4f}")
    
    # Evaluate CF
    print("\nEvaluating User-Based CF Model...")
    start_time = time.time()
    cf_preds = cf.test(testset)
    cf_pred_time = time.time() - start_time
    rmse_cf = accuracy.rmse(cf_preds, verbose=False)
    map_cf = map_at_k(cf_preds, k=10)
    print(f"CF RMSE: {rmse_cf:.4f}")
    print(f"CF MAP@10: {map_cf:.4f}")
    
    # Save evaluation results
    results = {
        'svd': {
            'rmse': rmse_svd,
            'map10': map_svd,
            'train_time': svd_train_time,
            'pred_time': svd_pred_time
        },
        'cf': {
            'rmse': rmse_cf,
            'map10': map_cf,
            'train_time': cf_train_time,
            'pred_time': cf_pred_time
        }
    }
    
    results_path = os.path.join(models_dir, 'results.joblib')
    joblib.dump(results, results_path)
    print(f"\nSaved results to {results_path}")
    
    # Print comparison table
    print("\n--- MODEL COMPARISON TABLE ---")
    print("| Dimension | SVD (Matrix Factorization) | User-Based Collaborative Filtering |")
    print("| --- | --- | --- |")
    print(f"| **Training Time** | {svd_train_time:.2f}s (Moderate) | {cf_train_time:.2f}s (Slow - pairwise similarity) |")
    print(f"| **Prediction Speed** | {svd_pred_time:.2f}s (Fast - dot product) | {cf_pred_time:.2f}s (Slow - KNN lookup) |")
    print("| **Memory Usage** | Low (factor matrices) | High (similarity matrix) |")
    print("| **Handles Sparsity** | Very Well | Poorly |")
    print("| **Explainability** | Low (latent factors) | High (similar users) |")
    print(f"| **RMSE (Lower is better)** | **{rmse_svd:.4f}** | {rmse_cf:.4f} |")
    print(f"| **MAP@10 (Higher is better)** | **{map_svd:.4f}** | {map_cf:.4f} |")

if __name__ == '__main__':
    run_evaluation()
