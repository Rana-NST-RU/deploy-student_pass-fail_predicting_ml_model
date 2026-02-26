import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score

#  Load dataset
df = pd.read_csv("data/student_performance.csv")
print(df.isnull().sum())
#  Remove outliers using IQR method

cols = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "total_score"
]

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Dataset size after outlier removal:", df.shape)

#  Convert grade → pass/fail
df["result"] = df["grade"].apply(lambda g: 1 if g in ["A", "B", "C"] else 0)

#  Select features
X = df[[
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "total_score"
]]

y = df["result"]

#  Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Logistic Regression (Prediction)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#  Predictions
y_pred = model.predict(X_test_scaled)

#  Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)

#  K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaler.fit_transform(X))

df["cluster"] = clusters

print("\nCluster counts:")
print(df["cluster"].value_counts())