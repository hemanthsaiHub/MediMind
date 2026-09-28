"""Train risk models and export them as portable JSON.

Runtime inference needs no scikit-learn (keeps the Vercel bundle tiny), so we
export the logistic-regression parameters. A random forest is trained too and
reported in metrics.json for the model-comparison view.

Data: if data/diabetes.csv (Kaggle Pima Indians Diabetes) exists it is used for
the diabetes model; everything else uses a synthetic generator so the project
runs out of the box. Replace with real public datasets for your report.
"""
import csv, json, math, sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.features import SPECS  # noqa: E402

OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
N = 6000


def synth_diabetes(rng):
    preg = rng.poisson(3.5, N).clip(0, 17)
    glucose = rng.normal(120, 30, N).clip(50, 250)
    bp = rng.normal(70, 12, N).clip(35, 120)
    bmi = rng.normal(32, 7, N).clip(15, 65)
    ped = rng.gamma(2, 0.25, N).clip(0.05, 2.4)
    age = (21 + rng.gamma(2, 8, N)).clip(21, 80)
    logit = -9.0 + 0.035 * glucose + 0.07 * bmi + 0.6 * ped + 0.03 * age + 0.06 * preg + 0.005 * bp
    y = (rng.random(N) < 1 / (1 + np.exp(-(logit + rng.normal(0, 0.5, N))))).astype(int)
    return np.column_stack([preg, glucose, bp, bmi, ped, age]), y


def pima_diabetes(path):
    X, y = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            g, b, m = float(r["Glucose"]), float(r["BloodPressure"]), float(r["BMI"])
            if g <= 0 or b <= 0 or m <= 0:  # zeros mean "missing" in this dataset
                continue
            X.append([float(r["Pregnancies"]), g, b, m, float(r["DiabetesPedigreeFunction"]), float(r["Age"])])
            y.append(int(r["Outcome"]))
    return np.array(X), np.array(y)


def synth_heart(rng):
    age = rng.normal(54, 9, N).clip(29, 77)
    sex = rng.binomial(1, 0.68, N)
    cp = rng.choice([0, 1, 2, 3], N, p=[0.25, 0.15, 0.25, 0.35])
    bp = rng.normal(131, 17, N).clip(90, 200)
    chol = rng.normal(246, 50, N).clip(120, 450)
    hr = (205 - 0.7 * age + rng.normal(0, 15, N)).clip(70, 205)
    ang = rng.binomial(1, 0.33, N)
    st = rng.exponential(1.0, N).clip(0, 6)
    logit = -3.6 + 0.05 * age + 0.9 * sex + 0.6 * cp + 0.02 * bp + 0.004 * chol - 0.03 * hr + 1.1 * ang + 0.6 * st
    y = (rng.random(N) < 1 / (1 + np.exp(-(logit + rng.normal(0, 0.5, N))))).astype(int)
    return np.column_stack([age, sex, cp, bp, chol, hr, ang, st]), y


def scores(model, Xte, yte):
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(yte, pred), 3),
        "precision": round(precision_score(yte, pred), 3),
        "recall": round(recall_score(yte, pred), 3),
        "f1": round(f1_score(yte, pred), 3),
        "roc_auc": round(roc_auc_score(yte, p), 3),
    }


def main():
    rng = np.random.default_rng(42)
    report = {}
    for disease in SPECS:
        source = "synthetic demo data"
        if disease == "diabetes" and (ROOT / "data" / "diabetes.csv").exists():
            X, y = pima_diabetes(ROOT / "data" / "diabetes.csv")
            source = "Pima Indians Diabetes (data/diabetes.csv)"
        else:
            X, y = synth_diabetes(rng) if disease == "diabetes" else synth_heart(rng)
        names = [f["name"] for f in SPECS[disease]["features"]]
        assert X.shape[1] == len(names)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1).fit(Xtr, ytr)

        sc, clf = lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
        params = {
            "features": names,
            "mean": [float(v) for v in sc.mean_],
            "std": [float(v) for v in sc.scale_],
            "coef": [float(v) for v in clf.coef_[0]],
            "intercept": float(clf.intercept_[0]),
        }
        # Sanity check: manual JSON-parameter inference must match scikit-learn.
        z = (Xte[0] - np.array(params["mean"])) / np.array(params["std"])
        manual = 1 / (1 + math.exp(-(params["intercept"] + float(np.dot(params["coef"], z)))))
        assert abs(manual - lr.predict_proba(Xte[:1])[0, 1]) < 1e-9

        (OUT / f"{disease}.json").write_text(json.dumps(params, indent=2))
        report[disease] = {
            "data_source": source,
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "positive_rate": round(float(y.mean()), 3),
            "served_model": "logistic_regression",
            "models": {"logistic_regression": scores(lr, Xte, yte), "random_forest": scores(rf, Xte, yte)},
        }
        print(disease, source, report[disease]["models"])
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
