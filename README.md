# Cineflex — a movie recommender 

This repository contains my submission for the CULT Open Projects 2026 (IIT Roorkee). I built a movie recommendation engine using the Netflix Prize dataset, implementing and comparing Matrix Factorization (SVD) and User-Based Collaborative Filtering (KNN). 

## 🎬 About Cineflex
**Cineflex** is a premium, interactive web application built with Streamlit that serves as a movie recommendation engine. It bridges recommendation systems theory with a polished, streaming-like user interface. 

### 🚀 Key Features
- **Interactive Recommendation Dashboard**: Browse through the top active database subscribers and view SVD recommendations vs User-Based Collaborative Filtering recommendations.
- **Real-time Onboarding (Cold-Start Quiz)**: Select your favorite genres and rate popular movies. The backend projects these ratings directly into the SVD latent factor space to yield immediate, personalized recommendations.
- **Explainable AI (XAI)**: Includes neighbor-lookup tracing for User-Based KNN, showing *why* a movie is recommended (i.e. highlighting similar users' ratings).
- **Movie Similarity Explorer**: Explore and rank movie relationships by calculating the cosine similarity of their 100-dimensional SVD item vectors.
- **Interactive EDA Tab**: Explore dataset insights (distributions, power-law curves, and matrix sparsity heatmaps) dynamically.

### 🎥 App Demonstration & Screenshot
Here is a screenshot of the Cineflex dashboard interface:

![Cineflex Dashboard Screenshot](cineflex_screenshot.png)

Check out the interactive demo in action:

<video src="cineflex_demo.mp4" width="100%" controls alt="Cineflex Demo Video"></video>

---

###  Personal Context & Struggles
- **Dataset Size & Timeouts**: The raw Netflix dataset is huge (~2GB). I initially tried downloading it programmatically inside the notebooks, but `gdown` kept timing out or getting blocked by Google Drive's large-file warnings. I solved this by pre-downloading the archive files and creating a preprocessing script to clean and filter them locally.
- **Model Training Speeds**: Training KNN on my machine was a massive pain — it took over 3 minutes to compute the cosine similarities and use them. Matrix Factorization (SVD) was a lifesaver, fitting in just 2 seconds while giving much better recommendations.

---

##  Repository Structure

*   `dashboard.py` — Streamlit app to explore recommendations and model stats interactively.
*   `data/`
    *   `download_and_prepare.py` — Preprocessing script to clean up ratings and format them.
*   `eda/`
    *   `eda.py` — Helper script to run EDA and dump rating/user stats and plots.
*   `models/`
    *   `train.py` — Simple script that splits the ratings 80/20, runs SVD/KNN fits, and saves the models.
*   `evaluate/`
    *   `evaluate.py` — Calculates RMSE and MAP@10 metrics.
*   `recommend/`
    *   `recommend.py` — CLI test script for checking recommendations and neighbor lists.
*   `Technical_Report.md` — The technical write-up of the math, models, and comparisons.
*   `Presentation.md` — Presentation outline (8 slides) summarizing the results.
*   `requirements.txt` — Packages you need to install.
*   `scratch/` — A messy directory with helper scripts, PDF conversion files, and test notebooks.

---

##  How to Run

### 1. Install dependencies
Set up a virtual environment and install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Preprocess data
This filters the raw dataset to keep active users (>=20 ratings) and popular movies (>=50 ratings):
```bash
python3 data/download_and_prepare.py
```

### 3. Generate EDA plots
Creates statistics and saves distribution plots into `eda/plots/`:
```bash
python3 eda/eda.py
```

### 4. Train the models
Run this to train both SVD and User-Based KNN models and serialize them to `./models/`:
```bash
python3 models/train.py
```

### 5. Run evaluation
 **Run this before the dashboard or it'll crash!** The dashboard expects `models/results.joblib` to load. If it's not there, the dashboard will fail or fall back to static values.
```bash
python3 evaluate/evaluate.py
```

### 6. Start the dashboard
Launch the Streamlit app:
```bash
streamlit run dashboard.py
```

---

##  Summary Results

*   **SVD Model**: RMSE = **0.9307** | MAP@10 = **73.59%** | Train Time = **2.01s** | Prediction Time = **1.29s**
*   **User-Based KNN**: RMSE = **0.9838** | MAP@10 = **70.67%** | Train Time = **220.55s** | Prediction Time = **93.50s**

**SVD clearly won** — it was both more accurate (lower RMSE and higher MAP@10) and over 40x faster than KNN during prediction. KNN also struggles because its user-user similarity matrix scales terribly in memory.
