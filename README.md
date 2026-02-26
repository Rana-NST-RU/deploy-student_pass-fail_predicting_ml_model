# 📊 Intelligent Learning Analytics Dashboard

A machine learning-powered Streamlit web application that analyzes student performance data, predicts pass/fail outcomes using Logistic Regression, and clusters students into performance groups using K-Means clustering.

---

## 🚀 Features

- **Batch Dataset Analysis** — Upload a CSV of student records to:
  - Predict pass/fail for each student using Logistic Regression
  - Cluster students into 3 performance groups using K-Means
  - Visualize score distributions, study hours vs. score, and attendance vs. score
  - Generate personalized improvement recommendations for at-risk students

- **Individual Student Prediction** — Enter parameters via interactive sliders to:
  - Get an instant pass/fail prediction
  - Receive targeted improvement suggestions based on weak areas

---

## 🗂️ Project Structure

```
learning-analytics-ml/
├── app.py                  # Main Streamlit dashboard (two-mode UI)
├── train_model.py          # Standalone model training & evaluation script
├── data/
│   └── student_performance.csv   # Student dataset (~28 MB, ~students records)
└── README.md
```

---

## 🧠 ML Models Used

| Model | Purpose |
|---|---|
| **Logistic Regression** | Binary classification — predicts pass (1) or fail (0) |
| **K-Means Clustering** | Unsupervised grouping of students into 3 performance clusters |
| **Standard Scaler** | Feature normalization before model training |

---

## 📋 Dataset

**File:** `data/student_performance.csv`

**Key columns used:**

| Column | Description |
|---|---|
| `student_id` | Unique student identifier |
| `weekly_self_study_hours` | Hours of self-study per week |
| `attendance_percentage` | Percentage of classes attended |
| `class_participation` | Participation score (0–10) |
| `total_score` | Overall academic score |
| `grade` | Letter grade (A, B, C, D, F) |

**Target variable:** `grade` is binarized → `pass` (A/B/C = 1, D/F = 0)

---

## ⚙️ How It Works

### Batch Dataset Analysis (`app.py`)

1. Upload a student CSV file
2. The app binarizes grades into pass/fail labels
3. Features (`weekly_self_study_hours`, `attendance_percentage`, `class_participation`) are scaled with `StandardScaler`
4. A **Logistic Regression** model is trained on an 80/20 train-test split
5. **K-Means** groups students into 3 clusters
6. Results are visualized with scatter plots and histograms
7. Students predicted to fail receive personalized recommendations

### Individual Prediction (`app.py`)

1. Adjust sliders for study hours, attendance %, and class participation
2. Click **Predict Result**
3. The app returns a **PASS ✅** or **FAIL ❌** prediction
4. On failure, specific improvement tips are displayed based on which thresholds are not met

### Standalone Training (`train_model.py`)

- Loads `data/student_performance.csv`
- Removes outliers via the **IQR method**
- Trains Logistic Regression and evaluates with **accuracy** and **precision** metrics
- Runs K-Means clustering and prints cluster distribution

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| **Python 3** | Core language |
| **Streamlit** | Interactive web dashboard |
| **Pandas** | Data loading & manipulation |
| **scikit-learn** | ML models (Logistic Regression, K-Means, StandardScaler) |
| **Matplotlib** | Plot rendering |
| **Seaborn** | Statistical visualizations |
| **NumPy** | Numerical computing |

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd learning-analytics-ml
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn numpy
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🏃 Running the Standalone Model Training Script

```bash
python train_model.py
```

This will output:
- Null value counts
- Dataset shape after outlier removal
- Model **accuracy** and **precision**
- Cluster distribution counts

---

## 📌 Thresholds Used for Recommendations

| Feature | Threshold for Concern |
|---|---|
| Weekly self-study hours | < 15 hours |
| Attendance percentage | < 75% |
| Class participation | < 5 / 10 |

---

## 📊 Visualizations (Batch Mode)

| Chart | Description |
|---|---|
| **Score Distribution** | Histogram + KDE of total scores |
| **Study Hours vs Score** | Scatter plot colored by cluster |
| **Attendance vs Score** | Scatter plot colored by cluster |

---

## 🔮 Future Improvements

- Save trained model to disk (e.g., with `joblib`) for reuse across sessions
- Add more features: assignment scores, quiz performance, engagement time
- Implement additional ML models (Random Forest, XGBoost) for comparison
- Add downloadable prediction reports (CSV/PDF)
- Deploy to Streamlit Cloud or HuggingFace Spaces

---

## 👤 Author

**Divyanshu Raj**  
Machine Learning · Data Analytics · Python Development
