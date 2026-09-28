from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
LOW = {"pregnancies": 0, "glucose": 85, "blood_pressure": 70, "bmi": 22, "pedigree": 0.1, "age": 25}
HIGH = {"pregnancies": 8, "glucose": 190, "blood_pressure": 90, "bmi": 42, "pedigree": 1.5, "age": 60}


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_schema_has_both_diseases():
    assert set(client.get("/api/schema").json()) == {"diabetes", "heart"}


def test_risk_ordering():
    lo = client.post("/api/predict", json={"disease": "diabetes", "values": LOW}).json()
    hi = client.post("/api/predict", json={"disease": "diabetes", "values": HIGH}).json()
    assert lo["risk_percent"] < 30 < 60 < hi["risk_percent"]
    assert hi["level"] == "High" and hi["factors"]


def test_validation_errors():
    bad = dict(LOW, glucose=9999)
    r = client.post("/api/predict", json={"disease": "diabetes", "values": bad})
    assert r.status_code == 422 and "Glucose" in r.json()["detail"]
    assert client.post("/api/predict", json={"disease": "flu", "values": {}}).status_code == 404


def test_forecast_demo_and_custom():
    d = client.get("/api/forecast/beds?days=5").json()
    assert len(d["forecast"]) == 5 and d["capacity"] == 200
    c = client.post("/api/forecast/beds", json={"history": [100 + i for i in range(20)], "days": 3}).json()
    assert c["forecast"][0]["value"] > 115
    assert client.post("/api/forecast/beds", json={"history": [1, 2, 3]}).status_code == 422


def test_assistant_offline(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert client.post("/api/assistant", json={"question": "What is BMI?"}).json()["mode"] == "offline"


def test_metrics_present():
    m = client.get("/api/metrics").json()
    assert m["diabetes"]["models"]["logistic_regression"]["roc_auc"] > 0.7
