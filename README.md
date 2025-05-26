# ⚽ Player Position Classifier

**Player Position Classifier** is a machine learning project that predicts football players' positions (DF, MF, FW, GK) using player performance statistics from the Premier League 2023/2024 season. It leverages Random Forest and pipeline-based preprocessing to deliver accurate classifications from either raw or cleaned datasets.

---

## 🚀 Features

- 🧠 Predict player positions using Random Forest
- 🧹 Clean and merge multi-sheet Excel datasets
- 🔍 Includes both raw and cleaned data pipelines
- 🧪 GridSearchCV optimization with evaluation reports
- 📊 Confusion matrix and feature importance visualizations
- 📦 Pre-trained `.pkl` pipelines for easy reuse

---

## 🧾 File Structure

```
📁 player-position-classifier
├── prepare_data.py
├── no_position_cleaning.py
├── testing.py
├── testing_no_cleaning_position.py
├── merging.py
├── player_position_pipeline.pkl
├── player_position_pipeline2.pkl
├── feature_columns.pkl
├── feature_columns2.pkl
├── merged_premier_league_stats.csv
├── Premier League Players 23_24 Stats_test.xlsx
├── confusion_matrix.png
├── feature_importances.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/player-position-classifier.git
cd player-position-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Run the Classifier

### ▶️ Using Cleaned Data:
```bash
python testing.py
```

### ▶️ Using Raw Data:
```bash
python testing_no_cleaning_position.py
```

> Ensure `Premier League Players 23_24 Stats_test.xlsx` is in the same folder.

---

## 📊 Visual Outputs

- `confusion_matrix.png` – true vs. predicted position visualization
- `feature_importances.png` – shows most important features in classification

---

## 📁 Data Sources

- Premier League 2023/24 player statistics (cleaned & merged)
- Feature engineering for position prediction

---

## 🔁 Model Pipelines

The classifier is saved as a `Pipeline` using `joblib`, so it can be loaded easily:

```python
import joblib
model = joblib.load('player_position_pipeline.pkl')
```

---

## 🪪 License

MIT © 2025 Sumaya Nasser Alhashmi

---

Ready to classify your own players? Just load the pipeline and predict! 🎯
