import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("beans_dataset.csv")
print("Rows loaded:", len(df))

df = df.replace("NaN", None).dropna()
df = df.drop_duplicates().reset_index(drop=True)
print("Rows after cleaning:", len(df))


print("\nBean counts by variety:")
print(df["Class"].value_counts())
print("\nAverage Area by variety:")
print(df.groupby("Class")["Area"].mean().round(1))
print("\nAverage roundness by variety:")
print(df.groupby("Class")["roundness"].mean().round(3))


fig1, ax1 = plt.subplots(figsize=(8, 4))
df["Class"].value_counts().plot(kind="bar", ax=ax1, color="tab:cyan", title="Bean count by variety")
ax1.set_ylabel("Count")
fig1.tight_layout()

fig2, ax2 = plt.subplots(figsize=(8, 4))
df.groupby("Class")["Area"].mean().plot(kind="bar", ax=ax2, color="tab:blue", title="Average Area by bean variety")
ax2.set_ylabel("Area")
fig2.tight_layout()

fig3, ax3 = plt.subplots(figsize=(8, 4))
df.groupby("Class")["roundness"].mean().plot(kind="bar", ax=ax3, color="tab:purple", title="Average roundness by bean variety")
ax3.set_ylabel("Roundness")
fig3.tight_layout()


X = df.drop(columns="Class").astype(float)
y = df["Class"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

nn = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=42)
nn.fit(X_train, y_train)


rf_acc = accuracy_score(y_test, rf.predict(X_test)) * 100
nn_acc = accuracy_score(y_test, nn.predict(X_test)) * 100
rf_f1 = f1_score(y_test, rf.predict(X_test), average="macro")
nn_f1 = f1_score(y_test, nn.predict(X_test), average="macro")

print("\n---- Results (higher accuracy is better) ----")
print(f"Random Forest : {rf_acc:.2f}% accuracy, Macro F1 = {rf_f1:.3f}")
print(f"Neural Network: {nn_acc:.2f}% accuracy, Macro F1 = {nn_f1:.3f}")

if rf_acc > nn_acc:
    winner = "Random Forest"
else:
    winner = "Neural Network"
print("Winner:", winner)

results = {"Random Forest": round(rf_acc, 2), "Neural Network": round(nn_acc, 2)}

fig4, ax4 = plt.subplots(figsize=(8, 4))
pd.Series(results).plot(kind="bar", ax=ax4, color=["tab:green", "tab:orange"],
                        title="Model Accuracy (%) (higher is better)")
ax4.set_ylim(0, 105)
ax4.set_ylabel("Accuracy (%)")
fig4.tight_layout()


if __name__ == "__main__":
    plt.show()
