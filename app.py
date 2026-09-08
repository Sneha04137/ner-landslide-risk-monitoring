from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained Random Forest model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "NER_RandomForest_Model.pkl"
)

model = joblib.load(MODEL_PATH)

# Exact feature order used during model training
FEATURES = [
    "latitude",
    "longitude",
    "rainfall_mm_day",
    "elevation_m",
    "slope_deg",
    "aspect_deg",
    "distance_to_river_m",
    "ndvi",
    "lulc_class"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        values = [
            float(data["latitude"]),
            float(data["longitude"]),
            float(data["rainfall_mm_day"]),
            float(data["elevation_m"]),
            float(data["slope_deg"]),
            float(data["aspect_deg"]),
            float(data["distance_to_river_m"]),
            float(data["ndvi"]),
            float(data["lulc_class"])
        ]

        X = np.array(values).reshape(1, -1)

        prediction = model.predict(X)[0]

        # Get prediction probability if available
        probabilities = model.predict_proba(X)[0]
        max_probability = float(np.max(probabilities) * 100)

        return jsonify({
            "success": True,
            "risk": str(prediction),
            "confidence": round(max_probability, 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/map")
def risk_map():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "NER_Landslide_Risk_Map.html"
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )

