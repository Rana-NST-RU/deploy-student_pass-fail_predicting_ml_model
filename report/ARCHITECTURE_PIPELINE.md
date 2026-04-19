# Architectural Data Pipeline
### Student Pass/Fail Prediction System — End-to-End Data Flow

---

```
┌─────────────────────────────────────────────────────────────┐
│                     📂 RAW INPUT DATA                       │
│              data/student_data.csv  (~1,000,000 rows)       │
│                                                             │
│  Features (9):                                              │
│    parental_education_level  (1–7, ordinal)                 │
│    daily_study_hours         (0–12 h/day)                   │
│    attendance_rate           (0.0–1.0)                      │
│    sleep_hours               (3–12 h/night)                 │
│    stress_level              (1–10)                         │
│    motivation_score          (0–100)                        │
│    math_score                (0–100)                        │
│    reading_score             (0–100)                        │
│    writing_score             (0–100)                        │
│                                                             │
│  Target: pass_fail  ("Pass" | "Fail")                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  🧹 STEP 1: NULL HANDLING                   │
│                 DataProcessor.fill_nulls()                  │
│                                                             │
│  For each of the 9 feature columns:                         │
│    df[col].fillna(df[col].median())                         │
│  → Missing values replaced with per-column median           │
│  → Result: 0 null values remain                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               📊 STEP 2: OUTLIER REMOVAL (IQR)              │
│              DataProcessor.remove_outliers()                │
│                                                             │
│  For each of the 9 feature columns:                         │
│    Lower Bound = Q1 − (1.5 × IQR)                           │
│    Upper Bound = Q3 + (1.5 × IQR)                           │
│  → Rows outside bounds are dropped                          │
│  → Columns with IQR = 0 are skipped                         │
│  → Reduces dataset by ~1–2%                                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              🏷️  STEP 3: TARGET ENCODING                    │
│               DataProcessor.preprocess()                    │
│                                                             │
│  df["result"] = 1 if pass_fail == "Pass" else 0             │
│                                                             │
│  Class distribution (approximate):                          │
│    Pass (1) ≈ 50–70%   |   Fail (0) ≈ 30–50%               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              📏 STEP 4: PERCENTILE CLIPPING + SCALING        │
│               DataProcessor.preprocess()                    │
│                                                             │
│  1. Clip each feature to [1st percentile, 99th percentile]  │
│     → Suppresses remaining extreme values before scaling    │
│                                                             │
│  2. MinMaxScaler().fit_transform(X)                         │
│     → All 9 features normalised to [0.0 → 1.0]              │
│     → scaler.pkl saved for inference                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               ✂️  STEP 5: TRAIN / TEST SPLIT                │
│             DataProcessor.split_and_balance()               │
│                                                             │
│  train_test_split(X_scaled, y,                              │
│                   test_size=0.2, random_state=42,           │
│                   stratify=y)                               │
│                                                             │
│  Training Set:  80%  (stratified — same Pass/Fail ratio)    │
│  Test Set:      20%  (held out, never touched by SMOTE)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              🔁 STEP 6: SMOTE OVERSAMPLING                  │
│             DataProcessor.split_and_balance()               │
│                  (applied to TRAIN SET only)                │
│                                                             │
│  SMOTE(random_state=42).fit_resample(X_train, y_train)      │
│  → Generates synthetic minority-class samples               │
│  → Balances Fail class to match Pass count in training set  │
│  → Prevents model from always predicting the majority class │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          🏆 STEP 7: MODEL SELECTION via GridSearchCV        │
│             ModelTrainer.train_and_evaluate()               │
│                                                             │
│  Two candidates, each tuned with 3-fold StratifiedKFold,   │
│  scored by weighted F1:                                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Logistic Regression  C ∈ {0.1, 1.0, 10.0}           │   │
│  │ XGBoost              n_estimators ∈ {50, 100}        │   │
│  │                      max_depth    ∈ {3, 6}           │   │
│  │                      learning_rate∈ {0.05, 0.1}      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Winner: model with highest weighted F1 on test set         │
│  → Saved as model.pkl                                       │
│  → Metrics (accuracy/precision/recall/F1) saved in meta.pkl │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            🔵 STEP 8: K-MEANS CLUSTERING                    │
│              ModelTrainer.train_kmeans()                    │
│                  (Unsupervised — no labels)                 │
│                                                             │
│  KMeans(n_clusters=3, random_state=42).fit(X_scaled)        │
│  → Segments students into 3 performance clusters:           │
│      Cluster 0: High Achievers                              │
│      Cluster 1: Average Performers                          │
│      Cluster 2: Struggling / At-Risk Students               │
│  → Saved as: kmeans.pkl                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               💾 STEP 9: ARTIFACT SERIALISATION             │
│              ModelTrainer.save_artifacts()                  │
│                                                             │
│  All artifacts saved to /models/ using joblib:              │
│                                                             │
│  ┌──────────────────────────┬───────────────────────────┐  │
│  │ File                     │ Purpose                   │  │
│  ├──────────────────────────┼───────────────────────────┤  │
│  │ model.pkl                │ Best trained classifier   │  │
│  │ scaler.pkl               │ Fitted MinMaxScaler       │  │
│  │ kmeans.pkl               │ Behavioural cluster model │  │
│  │ meta.pkl                 │ Model name + F1/acc/etc   │  │
│  │ feature_names.pkl        │ Ordered feature name list │  │
│  │ input_feature_cols.pkl   │ Feature column list       │  │
│  └──────────────────────────┴───────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│        🖥️  STEP 10: STREAMLIT DASHBOARD (app.py)            │
│                                                             │
│  PAGE 1 — Dashboard (pages/dashboard.py):                   │
│    1. User sets 9 sliders (scores, habits, wellbeing)       │
│    2. Input → DataFrame → scaler.transform()                │
│    3. model.predict_proba() → Pass/Fail + probability %     │
│    4. kmeans.predict() → Behavioural cluster (0, 1, 2)      │
│    5. Rule-based tips generated from threshold checks       │
│    6. CTA button → navigates to Study Coach chat            │
│                                                             │
│  PAGE 2 — AI Study Coach (pages/chat_interface.py):         │
│    1. LangGraph agent pipeline invoked per message:         │
│       Router → Diagnose → Plan → Retrieve → Respond → Memory│
│    2. Groq LLM (llama-3.1-8b-instant) generates responses   │
│    3. ChromaDB RAG retrieves relevant study tips            │
│    4. Tavily live web search adds current resource links    │
│    5. SessionMemory persists conversation history           │
│    6. Off-topic questions are politely declined             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     📤 FINAL OUTPUT                         │
│                                                             │
│  Dashboard:  Pass ✅ / Fail ❌ + Probability %              │
│              Behavioural Cluster (High / Average / At-Risk) │
│              Score breakdown bars (Math / Reading / Writing)│
│              Personalised actionable improvement tips       │
│                                                             │
│  Study Coach: Personalised 7-day study plan                 │
│               RAG-retrieved study tips                      │
│               Live web resource links (Tavily)              │
│               Conversational coaching (multi-turn)          │
└─────────────────────────────────────────────────────────────┘
```

---

**⚠️ Note on model metrics:** All evaluation metrics (accuracy, precision, recall, F1) reach
1.00 on this dataset because the `pass_fail` label in the synthetic CSV is a deterministic
function of the input feature scores (specifically the average of `math_score`,
`reading_score`, and `writing_score`). The classifier trivially learns this rule. On a
real-world dataset with independently collected labels and natural noise, metrics would be
lower and more representative.

---

**Team:** Divyanshu Raj · Yash Agarwal · Abhijeet Kumar · Ranajeet Roy
