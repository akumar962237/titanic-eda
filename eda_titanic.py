"""
Titanic Survival - Exploratory Data Analysis (EDA)
Steps: load -> inspect -> clean -> feature engineering -> analyze -> visualize
Run:  pip install pandas seaborn matplotlib   then   python eda_titanic.py
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # saves charts to files; works without a display
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")

# ---------- 1. LOAD ----------
df = sns.load_dataset("titanic")
print("Shape:", df.shape)
print(df.head(), "\n")

# ---------- 2. INSPECT ----------
print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0], "\n")
# NOTE: duplicated() flags ~107 rows, but this dataset has no name/ID column, so
# different passengers can look identical. They are NOT true duplicates -> we keep them.
print("Rows that look identical (kept, not true duplicates):", df.duplicated().sum(), "\n")

# ---------- 3. CLEAN ----------
df = df.drop(columns=["deck"])                       # ~77% missing, not usable
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
df["age"] = df.groupby(["pclass", "sex"])["age"].transform(lambda x: x.fillna(x.median()))
print("Missing values after cleaning:", df.isnull().sum().sum(), "\n")

# ---------- 4. FEATURE ENGINEERING ----------
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                         labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])

# ---------- 5. ANALYZE ----------
print("Overall survival rate: %.1f%%" % (df["survived"].mean() * 100))
for col in ["sex", "pclass", "age_group", "embark_town"]:
    print(f"\nSurvival rate by {col} (%):")
    print((df.groupby(col, observed=True)["survived"].mean() * 100).round(1))
print("\nSurvival rate by family size (%):")
print((df.groupby("family_size")["survived"].mean() * 100).round(1))
print("\nMedian fare by class:")
print(df.groupby("pclass")["fare"].median().round(2))

# ---------- 6. VISUALIZE ----------
def save(name):
    plt.tight_layout()
    plt.savefig(f"charts/{name}.png", dpi=120)
    plt.close()

sns.barplot(data=df, x="sex", y="survived", errorbar=None)
plt.title("Survival Rate by Gender"); plt.ylabel("Survival rate"); save("1_survival_by_gender")

sns.barplot(data=df, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival Rate by Class and Gender"); plt.ylabel("Survival rate"); save("2_survival_by_class_gender")

sns.barplot(data=df, x="age_group", y="survived", errorbar=None)
plt.title("Survival Rate by Age Group"); plt.ylabel("Survival rate"); save("3_survival_by_age_group")

sns.barplot(data=df, x="family_size", y="survived", errorbar=None)
plt.title("Survival Rate by Family Size"); plt.ylabel("Survival rate"); save("4_survival_by_family_size")

sns.boxplot(data=df, x="pclass", y="fare")
plt.yscale("log"); plt.title("Fare Distribution by Class (log scale)"); save("5_fare_by_class")

sns.heatmap(df[["survived", "pclass", "age", "fare", "family_size"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap"); save("6_correlation_heatmap")

print("\nDone! 6 charts saved in the charts/ folder.")
