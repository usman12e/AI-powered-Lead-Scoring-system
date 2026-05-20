# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Flask web server — form input → 8-feature prediction pipeline → score
#
# Feature pipeline at prediction time (must match training exactly):
#   6 raw form values
#     → label encode 3 categoricals (encoders.pkl)
#     → standard scale 6 features   (scaler.pkl)
#     → compute 2 engineered features and standardise (eng_params.pkl)
#     → 8-feature array → model.pkl → probability score
# ─────────────────────────────────────────────────────────────────────────────

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

print("Loading model and pipeline tools...")
model      = joblib.load('models/model.pkl')
scaler     = joblib.load('models/scaler.pkl')
encoders   = joblib.load('models/encoders.pkl')
eng_params = joblib.load('models/eng_params.pkl')
print("Ready → http://localhost:5000")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # ── Step 1: Read raw form values ──────────────────────────────────────
        total_visits  = float(data['total_visits'])
        time_on_site  = float(data['time_on_site'])
        page_views    = float(data['page_views'])
        lead_source   = data['lead_source']
        last_activity = data['last_activity']
        occupation    = data['occupation']

        # ── Step 2: Label encode 3 categoricals ──────────────────────────────
        def safe_encode(col, value):
            enc = encoders[col]
            return int(enc.transform([value])[0]) if value in enc.classes_ else 0

        lead_source_enc   = safe_encode('Lead Source',                     lead_source)
        last_activity_enc = safe_encode('Last Activity',                   last_activity)
        occupation_enc    = safe_encode('What is your current occupation', occupation)

        # ── Step 3: Build 6-feature array (must match FEATURES order) ─────────
        #   [0] TotalVisits
        #   [1] Total Time Spent on Website
        #   [2] Page Views Per Visit
        #   [3] Lead Source
        #   [4] Last Activity
        #   [5] What is your current occupation
        raw_6 = np.array([[
            total_visits,
            time_on_site,
            page_views,
            lead_source_enc,
            last_activity_enc,
            occupation_enc,
        ]])

        # ── Step 4: Scale the 6 features ─────────────────────────────────────
        scaled_6 = scaler.transform(raw_6)

        # ── Step 5: Compute 2 engineered features using saved params ──────────
        # Uses raw values (not scaled) for the formulas, then standardises
        # with the TRAINING mean/std saved in eng_params.pkl
        engagement       = total_visits * time_on_site
        eng_scaled       = (engagement - eng_params['engagement_mean']) / eng_params['engagement_std']

        ratio            = page_views / (time_on_site + 1)
        ratio_scaled     = (ratio - eng_params['ratio_mean']) / eng_params['ratio_std']

        # ── Step 6: Stack → 8-feature array ──────────────────────────────────
        #   [0..5] → 6 original scaled features
        #   [6]    → Engagement Score
        #   [7]    → Pages-per-Minute
        features_8 = np.column_stack([scaled_6, [[eng_scaled]], [[ratio_scaled]]])

        # ── Step 7: Predict ───────────────────────────────────────────────────
        probability = model.predict_proba(features_8)[0][1]
        score       = round(float(probability) * 100)

        # ── Step 8: Label ─────────────────────────────────────────────────────
        if score >= 75:
            label, color = "High Potential",     "#1D9E75"
        elif score >= 50:
            label, color = "Medium Potential",   "#BA7517"
        elif score >= 25:
            label, color = "Low Potential",      "#D85A30"
        else:
            label, color = "Very Low Potential", "#E24B4A"

        return jsonify({'success': True, 'score': score,
                        'label': label, 'color': color})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)