from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import pandas as pd
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


# --------------------------------------------------
# AI RISK PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        X = pd.DataFrame([{
            "latitude": float(data["latitude"]),
            "longitude": float(data["longitude"]),
            "rainfall_mm_day": float(data["rainfall_mm_day"]),
            "elevation_m": float(data["elevation_m"]),
            "slope_deg": float(data["slope_deg"]),
            "aspect_deg": float(data["aspect_deg"]),
            "distance_to_river_m": float(data["distance_to_river_m"]),
            "ndvi": float(data["ndvi"]),
            "lulc_class": float(data["lulc_class"])
        }], columns=FEATURES)

        # Prediction
        prediction = model.predict(X)[0]

        # Prediction probabilities
        probabilities = model.predict_proba(X)[0]
        max_probability = float(max(probabilities) * 100)

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


# --------------------------------------------------
# HISTORICAL NER RISK MAP
# --------------------------------------------------

@app.route("/map")
def risk_map():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "NER_Landslide_Risk_Map.html"
    )


# --------------------------------------------------
# CURRENT ANALYZED LOCATION MAP
# --------------------------------------------------

@app.route("/current-map")
def current_map():

    try:
        latitude = float(request.args.get("lat"))
        longitude = float(request.args.get("lon"))
        risk = request.args.get("risk", "Unknown")
        confidence = request.args.get("confidence", "")

        # Safety check for coordinates
        if not (-90 <= latitude <= 90):
            raise ValueError("Invalid latitude")

        if not (-180 <= longitude <= 180):
            raise ValueError("Invalid longitude")

        # Only allow known model classes
        allowed_risks = ["Low", "Moderate", "High", "Very High"]

        if risk not in allowed_risks:
            risk = "Unknown"

        # Colors for risk levels
        risk_colors = {
            "Low": "green",
            "Moderate": "orange",
            "High": "red",
            "Very High": "darkred",
            "Unknown": "gray"
        }

        color = risk_colors[risk]

        html = f"""
<!DOCTYPE html>
<html>
<head>

    <title>Current Landslide Risk</title>

    <meta charset="utf-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <link rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>

    <style>

        html, body {{
            height: 100%;
            margin: 0;
        }}

        #map {{
            height: 100%;
            width: 100%;
        }}

        .risk-title {{
            position: absolute;
            top: 15px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;

            background: white;
            padding: 12px 22px;

            border-radius: 8px;

            box-shadow: 0 2px 8px rgba(0,0,0,0.3);

            font-family: Arial, sans-serif;

            text-align: center;
        }}

        .risk-title h2 {{
            margin: 0 0 5px 0;
        }}

        .risk-value {{
            font-size: 22px;
            font-weight: bold;
            color: {color};
        }}

    </style>

</head>

<body>

<div class="risk-title">

    <h2>📍 Current Landslide Risk</h2>

    <div class="risk-value">
        {risk}
    </div>

    <div>
        Confidence: {confidence}%
    </div>

</div>

<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>

    // Create map
    var map = L.map('map').setView(
        [{latitude}, {longitude}],
        10
    );

    // OpenStreetMap
    L.tileLayer(
        'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
        {{
            maxZoom: 19,
            attribution: '&copy; OpenStreetMap contributors'
        }}
    ).addTo(map);

    // Current analyzed location
    var marker = L.circleMarker(
        [{latitude}, {longitude}],
        {{
            radius: 14,
            color: '{color}',
            fillColor: '{color}',
            fillOpacity: 0.85,
            weight: 4
        }}
    ).addTo(map);

    // Popup
    marker.bindPopup(
        '<b>📍 Analyzed Location</b><br><br>' +
        '<b>Risk:</b> {risk}<br>' +
        '<b>Confidence:</b> {confidence}%<br>' +
        '<b>Latitude:</b> {latitude}<br>' +
        '<b>Longitude:</b> {longitude}'
    ).openPopup();

</script>

</body>
</html>
"""

        return html

    except Exception as e:

        return f"""
        <h2>Error creating risk map</h2>
        <p>{str(e)}</p>
        """, 400


# --------------------------------------------------
# RUN FLASK
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
