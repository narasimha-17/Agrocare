from datetime import datetime
from random import random
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_babel import Babel, _
import google.generativeai as genai
import os
import pickle
import numpy as np
from openai import models
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder , StandardScaler
from torchvision import transforms
import torch
from PIL import Image
import os
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
import yfinance as yf



app = Flask(__name__)
CORS(app)

app = Flask(__name__)


app.secret_key = "buddenarasimhasuryateja17042006" 
# ✅ Load your Gemini API key (replace with your key)
# GOOGLE_API_KEY = "AIzaSyDCToalcS0jGdZyNiFxRnJOnDkoWCYd6zA"
genai.configure(api_key="AIzaSyDCToalcS0jGdZyNiFxRnJOnDkoWCYd6zA")
model = genai.GenerativeModel("gemini-2.5-flash")


# Babel config
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
# Translation files location
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

babel = Babel(app)

# New v3+ style for language selection


# ✅ Home/index route
@app.route("/")
def index():
    return render_template("index.html")


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def clear_upload_folder(folder_path):
    """Remove all files from the upload folder."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

# Load the model
with open('trained models/plant_disease_btvgg16model.pkl', 'rb') as f:
    plant_disease_model = pickle.load(f)
# Define image size and labels (either hardcode or extract from dataset)
IMAGE_SIZE = 128
LABELS = [
    "Healthy",
    "Powdery",
    "Rust",
    ]





CSV_PATH = r"Datasets/synthetic_pest_pesticide_expanded_imbalanced.csv"
pesticide_df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')


# Use the correct absolute path to your CSV
"""CSV_PATH = r"Datasets/synthetic_pest_pesticide_expanded_imbalanced.csv"
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
# Example: rename column if actual name is different
df.rename(columns={'pestname ': 'PestName'}, inplace=True)
print(df.columns.tolist())
"""

# utf-8-sig removes BOM if present

    

DATASET_PATH = "Datasets/ai_crop_rotation_planner (2).xlsx"

#rotation_data = pd.read_excel(DATASET_PATH, engine='openpyxl')

# --- Load your trained machine learning model ---
MODEL_PATH = "trained models/best_crop_rotation_model.pkl"
model = joblib.load(MODEL_PATH)



soil_model = joblib.load("trained models/best_soil_health_model.pkl")
soil_model_features = pickle.load(open("trained models/soil_features.pkl", "rb"))






import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load training data
rotation_data = pd.read_excel(DATASET_PATH, engine='openpyxl')
categorical_features = ['Soil Type', 'Region', 'Season', 'Last Grown Crop']
target = 'Suggested Rotation Crop'

# Label encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    rotation_data[col] = le.fit_transform(rotation_data[col])
    label_encoders[col] = le

# Train model
X = rotation_data[categorical_features]
y = rotation_data[target]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "trained models/best_crop_rotation_model.pkl")
joblib.dump(label_encoders, "trained models/label_encoders.pkl")





# ✅ Detect Diseases page
'''@app.route("/detect-disease", methods=["GET", "POST"])
def detect_disease():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            # Clear old images
            clear_upload_folder(app.config['UPLOAD_FOLDER'])

            # ✅ Ensure the folder exists (in case it was deleted)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # this line caused the crash before

            # Preprocess image
            img = load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            class_index = np.argmax(preds)
            prediction = LABELS[class_index]
            image_url = filename

    return render_template("detect_disease.html", prediction=prediction, image_url=image_url)
'''
@app.route("/detect-disease", methods=["GET", "POST"])
def detect_disease():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            clear_upload_folder(app.config['UPLOAD_FOLDER'])
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = plant_disease_model.predict(img_array)
            class_index = np.argmax(preds)
            prediction = LABELS[class_index]
            image_url = filename

    return render_template("detect_disease.html", prediction=prediction, image_url=image_url)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ✅ Agro Services page
@app.route("/agro-services")
def agro_services():
    return render_template("agro_services.html")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load and fit scaler for crop recommendation (replace with your actual scaler/model loading as needed)
scaler = joblib.load("trained models/scaler_crop_recommendation.pkl")
model2 = joblib.load("trained models/crop_recommendation.pkl")
# Fit scaler with training data (replace with your actual training data path)
try:
    crop_df = pd.read_csv("Datasets/Crop_recommendation.csv")  # Update path as needed
    scaler.fit(crop_df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]])
except Exception as e:
    print(f"Warning: Could not fit scaler: {e}")

@app.route("/crop-recommendation", methods=["GET", "POST"])
def crop_recommendation():
    prediction = None
    if request.method == "POST":
        try:
            # Collect form data
            N = float(request.form["nitrogen"])
            P = float(request.form["phosphorus"])
            K = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # Create input array
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

            # Scale the input
            scaled_input = scaler.transform(input_data)

            # Make prediction
            result = model2.predict(scaled_input)[0]

            prediction = result

        except Exception as e:
            prediction = f"❌ Error: {str(e)}"

    return render_template("crop_recommendation.html", prediction=prediction)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Load label encoders and model (you should save and load encoders similarly)
with open('trained models/xgboost_model.pkl', 'rb') as file:
    model1 = pickle.load(file)

# Example label encoders — these should be saved and loaded from your training script
from sklearn.preprocessing import LabelEncoder 
state_encoder = LabelEncoder()
district_encoder = LabelEncoder()
market_encoder = LabelEncoder()
commodity_encoder = LabelEncoder()
variety_encoder = LabelEncoder()

# Fit them with the same data as in training (you should ideally load them)
df = pd.read_csv("Datasets/Price_Agriculture_commodities_Week.csv")  # Replace with actual path
state_encoder.fit(df['State'])
district_encoder.fit(df['District'])
market_encoder.fit(df['Market'])
commodity_encoder.fit(df['Commodity'])
variety_encoder.fit(df['Variety'])

@app.route("/price-commodity", methods=["GET", "POST"])
def price_commodity():
    prediction = None

    if request.method == "POST":
        try:
            state = request.form.get("state")
            district = request.form.get("district")
            market = request.form.get("market")
            commodity = request.form.get("commodity")
            variety = request.form.get("variety")

            # Encode inputs
            input_data = [
                state_encoder.transform([state])[0],
                district_encoder.transform([district])[0],
                market_encoder.transform([market])[0],
                commodity_encoder.transform([commodity])[0],
                variety_encoder.transform([variety])[0]
            ]

            # Predict modal price
            predicted_price = model1.predict([input_data])[0]
            prediction = round(predicted_price, 2)

            print("Prediction:", prediction)


        except Exception as e:
            prediction = f"❌ Error during prediction: {str(e)}"

    return render_template("price_commodity.html", prediction=prediction)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



@app.route("/crop_rotation", methods=["GET", "POST"])
def crop_rotation():
    suggestion = None

    if request.method == "POST":
        soil = request.form["soil"]
        region = request.form["region"]
        season = request.form["season"]
        last_crop = request.form["last_crop"]

        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            "Soil Type": soil,
            "Region": region,
            "Season": season,
            "Last Grown Crop": last_crop
        }])

        # Transform input using label encoders, ignore unknown categories
        transformed_df = input_df.copy()
        for col, le in label_encoders.items():
            # Map unknown categories to a default (e.g., first label)
            transformed_df[col] = transformed_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

        # Predict
        try:
            predicted_crop = model.predict(transformed_df)[0]
            suggestion = (
                f"For {soil} soil in {region} during {season}, "
                f"after {last_crop}, the best rotation crop is:{predicted_crop}."
            )
        except Exception as e:
            suggestion = f"⚠️ Model error: {e}"

    rotation_records = rotation_data.to_dict(orient="records")

    return render_template(
        "crop_rotation.html",
        suggestion=suggestion,
        
    )


# ✅ Pesticide Suggestion page



@app.route("/pesticide", methods=["GET", "POST"])
def pesticide():
    """
    Handles pest and crop selection form and shows pesticide recommendation.
    Keeps dropdown selections and result visible after submission.
    """

    try:
        pests = sorted(pesticide_df['PestName'].unique())
        hostcrops = sorted(pesticide_df['HostCrop'].unique())
    except KeyError as e:
        print(f"ERROR: Column not found. Details: {e}")
        pests = ["Error loading data"]
        hostcrops = ["Error loading data"]

    prediction = None
    selected_pest = None
    selected_crop = None

    if request.method == "POST":
        selected_pest = request.form.get("pest")
        selected_crop = request.form.get("hostcrop")

        # Filter based on selections
        result = pesticide_df[
            (pesticide_df['PestName'] == selected_pest) &
            (pesticide_df['HostCrop'] == selected_crop)
        ]

        if not result.empty:
            prediction = {
                "pesticide": result.iloc[0]['RecommendedPesticide'],
                "method": result.iloc[0]['ApplicationMethod']
            }
        else:
            prediction = {"pesticide": "No recommendation found", "method": "-"}

    # Pass selections and result back to template
    return render_template(
        "pesticide.html",
        pests=pests,
        hostcrops=hostcrops,
        prediction=prediction,
        selected_pest=selected_pest,
        selected_crop=selected_crop
    )

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


@app.route("/soil-prediction", methods=["GET", "POST"])
def soil_health():
    soil_prediction = None

    if request.method == "POST":
        try:
            # Collect form data
            form_data = {
                "SoilType": request.form["SoilType"],
                "pH_Range": request.form["pH_Range"],
                "OC_Level": request.form["OC_Level"],
                "N_Level": request.form["N_Level"],
                "P_Level": request.form["P_Level"],
                "K_Level": request.form["K_Level"]
            }

            input_df = pd.DataFrame([form_data])

            # One-hot encode
            input_encoded = pd.get_dummies(input_df)

            # Reindex to match training columns
            input_encoded = input_encoded.reindex(columns=soil_model_features, fill_value=0)

            # Predict
            prediction = soil_model.predict(input_encoded)[0]
            soil_prediction = f"🌱 Predicted Soil is Health: {prediction}"

        except Exception as e:
            soil_prediction = f"⚠️ Model error: {e}"

    return render_template("soilhealthprediction.html", soil_prediction=soil_prediction)







#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ✅ Weather Suggestion page
@app.route("/weather", methods=["GET", "POST"])
def weather():
    weather = None
    city = None

    if request.method == "POST":
        city = request.form.get("city")
        API_KEY = "dd05d85b1397672526d14d00a6903e9f"  # 🔑 replace with your OpenWeatherMap API key
        URL = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

        try:
            response = requests.get(URL).json()
            weather = {
                "temperature": response["main"]["temp"],
                "humidity": response["main"]["humidity"],
                "condition": response["weather"][0]["description"]
            }
        except Exception as e:
            weather = {
                "temperature": "N/A",
                "humidity": "N/A",
                "condition": "Location not found"
            }

    return render_template("weather.html", weather=weather, city=city)





# ✅ Contact page
@app.route("/contact")
def contact():
    return render_template("contact.html")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ✅ Chatbot route (Gemini API)
@app.route("/ask", methods=["POST"])
def ask():
    try:
        # 🔹 Parse and validate the request JSON
        data = request.get_json()
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"reply": "❌ Message is missing!"}), 400

        # 🔹 Generate response from Gemini API
        response = model.generate_content(message)

        # 🔹 Send reply back to frontend
        return jsonify({"reply": response.text})

    except Exception as e:
        # 🔹 Log the error for debugging
        print("❌ Internal Error:", str(e))
        return jsonify({"reply": f"❌ Error from Gemini: {str(e)}"}), 500
    





# --- Subsidy page ---
@app.route('/subsidy')
def subsidy():
    return render_template("subsidy.html")

# --- Form Handling ---
@app.route('/get_subsidy', methods=['POST'])
def get_subsidy():
    crop = request.form['crop']
    need = request.form['need']

    # Dummy recommendations (replace with ML model or DB later)
    recommendations = {
        ("wheat", "fertilizer"): "Government Wheat Fertilizer Subsidy - 30% off",
        ("rice", "irrigation"): "PM-KUSUM Solar Irrigation Subsidy - 40% off",
        ("maize", "equipment"): "Farm Machinery Loan @ 4% interest",
        ("cotton", "loan"): "Agri Loan Scheme - ₹2 Lakh at 3% interest"
    }

    subsidy = recommendations.get((crop, need), "No specific subsidy found. Check local govt portal.")

    return render_template("subsidy.html", subsidy=subsidy)

@app.route("/crop_guide")
def crop_guide():
    return render_template("cropGuide.html")





#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





RAPIDAPI_HOST = "commodity-prices2.p.rapidapi.com"
RAPIDAPI_KEY = "9d54b6cafbmsh83b472d22333b49p143b32jsn65b8146fd4b7"   # 🔑 replace with your key



COMMODITY_TICKERS = {
    "wheat": "ZW=F",
    "corn": "ZC=F",
    "soybean": "ZS=F",
    "rice": "ZR=F",
    "cotton": "CT=F",
    "coffee": "KC=F",
    "sugar": "SB=F",
    "cocoa": "CC=F",
    "orange_juice": "OJ=F"
}

@app.route("/api/commodity/<name>", methods=["GET"])
def get_commodity(name):
    try:
        ticker_symbol = COMMODITY_TICKERS.get(name.lower())
        if not ticker_symbol:
            return jsonify({"error": f"Commodity '{name}' not supported"}), 400

        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="5d", interval="1d")

        if data.empty:
            return jsonify({
                "commodity": name,
                "price": round(random.uniform(50, 500), 2),
                "unit": "USD",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "note": "Demo price (no live data available)"
            })

        last_quote = data.tail(1).iloc[0]
        timestamp = str(last_quote.name)  # use actual index timestamp

        return jsonify({
            "commodity": name,
            "price": round(last_quote["Close"], 2),
            "unit": "USD",
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({"error": f"Request failed: {str(e)}"}), 500


# Example translations dictionary for demonstration
TRANSLATIONS = {
    "en": "Commodity Price Tracking",
    "hi": "वस्तु मूल्य ट्रैकिंग",
    "te": "వస్తువు ధర ట్రాకింగ్"
}

@app.route("/price_tracking")
def price_tracking():
    lang = session.get("lang", "en")
    text = TRANSLATIONS.get(lang, TRANSLATIONS["en"])
    return render_template("tracking.html", text=text)


























if __name__ == "__main__":
    app.run(debug=True)
