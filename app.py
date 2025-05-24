from flask import Flask, render_template, request, jsonify
from datetime import datetime
import numpy as np
import joblib  # for loading the trained model

app = Flask(__name__)
model = joblib.load('xgb_heart_readmission_model.pkl')  # Ensure this file exists in the same directory

# ---------------------- Encoding Functions ----------------------

def encode_admission_type(admission_type):
    return {
        "admission_type_EMERGENCY": int(admission_type.upper() == "EMERGENCY"),
        "admission_type_URGENT": int(admission_type.upper() == "URGENT")
    }

def encode_flag(flag_value: str) -> int:
    mapping = {
        "nan": 0,
        "abnormal": 1,
        "delta": 2
    }
    return mapping.get(flag_value.lower(), 0)

def encode_discharge_location(value: str) -> dict:
    value = value.strip().upper()
    return {
        "discharge_location_HOME": int(value == "HOME"),
        "discharge_location_HOME HEALTH CARE": int(value == "HOME HEALTH CARE"),
        "discharge_location_SNF": int(value == "SNF"),
        "discharge_location_SHORT TERM HOSPITAL": int(value == "SHORT TERM HOSPITAL"),
        "discharge_location_REHAB/DISTINCT PART HOSP": int(value == "REHAB/DISTINCT PART HOSP"),
        "discharge_location_OTHER FACILITY": int(value == "OTHER FACILITY"),
    }

def get_insurance_risk(insurance_type: str) -> int:
    insurance_risk = {
        'Medicare': 3,
        'Medicaid': 4,
        'Private': 1,
        'Self Pay': 2,
        'Government': 2
    }
    return insurance_risk.get(insurance_type, 0)

def calculate_length_of_stay(admit_time_str: str, discharge_time_str: str) -> int:
    admit_time = datetime.strptime(admit_time_str, "%Y-%m-%d %H:%M:%S")
    discharge_time = datetime.strptime(discharge_time_str, "%Y-%m-%d %H:%M:%S")
    return (discharge_time - admit_time).days

def get_admit_weekday(admit_time_str: str) -> int:
    admit_time = datetime.strptime(admit_time_str, "%Y-%m-%d %H:%M:%S")
    return admit_time.weekday()

# ---------------------- Routes ----------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form

    # Get numeric lab values
    ntprobnp = float(data.get('ntprobnp'))
    creatinine = float(data.get('creatinine'))
    urea_nitrogen = float(data.get('urea_nitrogen'))
    sodium = float(data.get('sodium'))
    potassium = float(data.get('potassium'))
    albumin = float(data.get('albumin'))
    crp = float(data.get('c_reactive_protein'))
    hemoglobin = float(data.get('hemoglobin'))
    hematocrit = float(data.get('hematocrit'))
    magnesium = float(data.get('magnesium'))

    # Flags
    flags = {
        "ntprobnp_flag": encode_flag(data.get("ntprobnp_flag")),
        "creatinine_flag": encode_flag(data.get("creatinine_flag")),
        "urea nitrogen_flag": encode_flag(data.get("urea_nitrogen_flag")),
        "sodium_flag": encode_flag(data.get("sodium_flag")),
        "potassium_flag": encode_flag(data.get("potassium_flag")),
        "albumin_flag": encode_flag(data.get("albumin_flag")),
        "c-reactive protein_flag": encode_flag(data.get("c_reactive_protein_flag")),
        "hemoglobin_flag": encode_flag(data.get("hemoglobin_flag")),
        "hematocrit_flag": encode_flag(data.get("hematocrit_flag")),
        "magnesium_flag": encode_flag(data.get("magnesium_flag")),
    }

    # Admission Type
    admission = encode_admission_type(data.get("admission_type"))

    # Discharge Location
    discharge = encode_discharge_location(data.get("discharge_location"))

    # Insurance Risk
    insurance_risk = get_insurance_risk(data.get("insurance"))

    # Time-based features
    length_of_stay = calculate_length_of_stay(data.get("admit_time"), data.get("discharge_time"))
    admit_weekday = get_admit_weekday(data.get("admit_time"))

    # Final input dictionary
    result = {
        "ntprobnp": ntprobnp,
        "creatinine": creatinine,
        "urea_nitrogen": urea_nitrogen,
        "sodium": sodium,
        "potassium": potassium,
        "albumin": albumin,
        "c_reactive_protein": crp,
        "hemoglobin": hemoglobin,
        "hematocrit": hematocrit,
        "magnesium": magnesium,
        **flags,
        **admission,
        **discharge,
        "insurance": data.get("insurance"),
        "insurance_risk": insurance_risk,
        "length_of_stay": length_of_stay,
        "admit_weekday": admit_weekday,
        "admission_type": data.get("admission_type"),
        "discharge_location": data.get("discharge_location"),
        "admit_time": data.get("admit_time"),
        "discharge_time": data.get("discharge_time")
    }

    # Match the order your model expects
    feature_order = [
        'ntprobnp', 'ntprobnp_flag', 'creatinine', 'creatinine_flag', 'urea nitrogen', 'urea nitrogen_flag',
        'sodium', 'sodium_flag', 'potassium', 'potassium_flag', 'albumin', 'albumin_flag',
        'c-reactive protein', 'c-reactive protein_flag', 'hemoglobin', 'hemoglobin_flag',
        'hematocrit', 'hematocrit_flag', 'magnesium', 'magnesium_flag',
        'admission_type_EMERGENCY', 'admission_type_URGENT',
        'discharge_location_HOME', 'discharge_location_HOME HEALTH CARE', 'discharge_location_SNF',
        'discharge_location_SHORT TERM HOSPITAL', 'discharge_location_REHAB/DISTINCT PART HOSP',
        'discharge_location_OTHER FACILITY', 'insurance_risk', 'length_of_stay', 'admit_weekday'
    ]

    # Construct feature vector for prediction
    X = np.array([result.get(f, 0) for f in feature_order]).reshape(1, -1)

    # Run prediction
    prediction = model.predict(X)[0]

    # Render results template with all data
    return render_template('results.html', 
                         prediction=prediction,
                         input_data=result,
                         input_values=data)

if __name__ == '__main__':
    app.run(debug=True)