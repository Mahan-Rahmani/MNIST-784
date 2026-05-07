import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# خواندن CSV
df = pd.read_csv("MNIST-10000-784.csv")

# جدا کردن ویژگی‌ها و برچسب
X = df.drop("label", axis=1).values
y = df["label"].values

# تقسیم train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ساخت مدل
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        C=0.5,
        solver="saga"
    )
)

# آموزش
model.fit(X_train, y_train)

# 6) تست
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
