"""Feature definitions shared by training and the API.
The API builds its input form and validation from this file, so a new feature
only needs to be added here (then retrain)."""

SPECS = {
    "diabetes": {
        "title": "Diabetes risk",
        "features": [
            {"name": "pregnancies", "label": "Pregnancies", "min": 0, "max": 20, "default": 1, "step": 1},
            {"name": "glucose", "label": "Glucose (mg/dL)", "min": 40, "max": 300, "default": 110, "step": 1},
            {"name": "blood_pressure", "label": "Diastolic blood pressure (mm Hg)", "min": 30, "max": 140, "default": 72, "step": 1},
            {"name": "bmi", "label": "BMI", "min": 10, "max": 70, "default": 28, "step": 0.1},
            {"name": "pedigree", "label": "Family history score", "min": 0, "max": 2.5, "default": 0.4, "step": 0.01},
            {"name": "age", "label": "Age (years)", "min": 18, "max": 90, "default": 35, "step": 1},
        ],
    },
    "heart": {
        "title": "Heart disease risk",
        "features": [
            {"name": "age", "label": "Age (years)", "min": 20, "max": 90, "default": 50, "step": 1},
            {"name": "sex", "label": "Sex", "min": 0, "max": 1, "default": 1, "options": [[0, "Female"], [1, "Male"]]},
            {"name": "chest_pain", "label": "Chest pain type", "min": 0, "max": 3, "default": 0,
             "options": [[0, "None"], [1, "Atypical angina"], [2, "Non-anginal pain"], [3, "Typical angina"]]},
            {"name": "resting_bp", "label": "Resting blood pressure (mm Hg)", "min": 80, "max": 220, "default": 125, "step": 1},
            {"name": "cholesterol", "label": "Cholesterol (mg/dL)", "min": 100, "max": 600, "default": 210, "step": 1},
            {"name": "max_heart_rate", "label": "Max heart rate (bpm)", "min": 60, "max": 220, "default": 160, "step": 1},
            {"name": "exercise_angina", "label": "Chest pain during exercise", "min": 0, "max": 1, "default": 0,
             "options": [[0, "No"], [1, "Yes"]]},
            {"name": "st_depression", "label": "ST depression (ECG)", "min": 0, "max": 7, "default": 0.5, "step": 0.1},
        ],
    },
}
