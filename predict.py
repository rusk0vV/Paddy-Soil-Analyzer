from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Models"
NPK_MODEL_FILE = MODEL_DIR / "npk_model.pkl"
PH_MODEL_FILE = MODEL_DIR / "ph_model.pkl"
EC_MODEL_FILE = MODEL_DIR / "ec_model.pkl"
ORP_MODEL_FILE = MODEL_DIR / "orp_model.pkl"

N_RANGE = (0, 100)
P_RANGE = (0, 70)
K_RANGE = (0, 55)
PH_RANGE = (3.5, 9.0)
EC_RANGE = (0, 3500)
ORP_RANGE = (-350, 350)


npk_model = joblib.load(NPK_MODEL_FILE)
ph_model = joblib.load(PH_MODEL_FILE)
ec_model = joblib.load(EC_MODEL_FILE)
orp_model = joblib.load(ORP_MODEL_FILE)


def validate_range(name, value, minimum, maximum):
    if value < minimum or value > maximum:
        raise ValueError(
            f"{name}={value} is out of range. Valid range is {minimum} to {maximum}."
        )


def get_phase1_health_status(N, P, K, ph, EC):
    """Estimate Phase 1 soil health statuses from dry-soil sensor values."""
    nutrient_low = N < 35 or P < 18 or K < 12
    nutrient_critical = N < 20 or P < 10 or K < 8
    ph_stress = ph < 5.5 or ph > 7.5
    ph_critical = ph < 5.0 or ph > 8.3
    ec_low = EC < 250
    ec_high = EC > 2200

    statuses = []

    if ec_high:
        statuses.append("High Salinity")
    if ec_low:
        statuses.append("Low EC")

    if nutrient_critical:
        statuses.append("Nutrient Deficient")
    elif nutrient_low:
        statuses.append("Moderate Nutrient Stress")

    if ph_critical:
        statuses.append("Critical pH Stress")
    elif ph_stress:
        statuses.append("pH Stress")

    if not statuses:
        statuses.append("Healthy for Transplanting")

    return statuses


def get_phase2_health_status(ORP):
    """Estimate Phase 2 muddy-soil health statuses from ORP values."""
    statuses = []

    if ORP > 200:
        statuses.append("Oxidizing Stress (High ORP)")
        statuses.append("Flooding Required")
    elif ORP < -200:
        statuses.append("Reducing Stress (Low ORP)")
    elif -150 <= ORP <= 150:
        statuses.append("Healthy Redox for Paddy")
    else:
        statuses.append("Monitor Redox")

    return statuses


def phase1_predict(N, P, K, ph, EC):
    """Run Phase 1 predictions for NPK, pH, and EC inputs."""
    validate_range("N", N, *N_RANGE)
    validate_range("P", P, *P_RANGE)
    validate_range("K", K, *K_RANGE)
    validate_range("ph", ph, *PH_RANGE)
    validate_range("EC_uS_cm", EC, *EC_RANGE)

    npk_input = pd.DataFrame([[N, P, K]], columns=["N", "P", "K"])
    # Backward-compatible inference for old (ph-only) and new (ph+EC) PH models.
    ph_feature_count = None
    if hasattr(ph_model, "estimators_") and ph_model.estimators_:
        ph_feature_count = getattr(ph_model.estimators_[0], "n_features_in_", None)

    if ph_feature_count == 2:
        ph_input = pd.DataFrame([[ph, EC]], columns=["ph", "EC_uS_cm"])
    else:
        ph_input = pd.DataFrame([[ph]], columns=["ph"])

    ec_input = pd.DataFrame([[EC]], columns=["EC_uS_cm"])

    npk_prediction = npk_model.predict(npk_input)[0]
    ph_prediction = ph_model.predict(ph_input)[0]
    ec_prediction = ec_model.predict(ec_input)[0]

    lime_kg = max(0.0, float(ph_prediction[0]))
    gypsum_kg = max(0.0, float(ph_prediction[1]))

    ph_recommendation = {
        "Lime_kg_per_acre": lime_kg,
        "Gypsum_kg_per_acre": gypsum_kg,
    }

    ec_boost_kg = float(ec_prediction[0])
    ec_flush_liters = float(ec_prediction[1])

    if ec_boost_kg >= ec_flush_liters and ec_boost_kg > 0:
        ec_recommendation = {
            "Low_EC_Fertilizer_Boost_kg": ec_boost_kg,
            "Phase1_EC_Flush_Water_Liters": 0.0,
        }
    elif ec_flush_liters > 0:
        ec_recommendation = {
            "Low_EC_Fertilizer_Boost_kg": 0.0,
            "Phase1_EC_Flush_Water_Liters": ec_flush_liters,
        }
    else:
        ec_recommendation = {
            "Low_EC_Fertilizer_Boost_kg": 0.0,
            "Phase1_EC_Flush_Water_Liters": 0.0,
        }

    npk_recommendation = {
        "Urea_kg_per_acre": float(npk_prediction[0]),
        "DAP_kg_per_acre": float(npk_prediction[1]),
        "MOP_kg_per_acre": float(npk_prediction[2]),
    }

    health_statuses = get_phase1_health_status(N, P, K, ph, EC)

    # Keep health labels consistent with model-prescribed interventions.
    fertilizer_needed = any(value > 0 for value in npk_recommendation.values())
    ph_amendment_needed = (lime_kg > 0) or (gypsum_kg > 0)
    ec_intervention_needed = (ec_recommendation["Low_EC_Fertilizer_Boost_kg"] > 0) or (
        ec_recommendation["Phase1_EC_Flush_Water_Liters"] > 0
    )

    if fertilizer_needed and "Nutrient Replenishment Needed" not in health_statuses:
        health_statuses.append("Nutrient Replenishment Needed")
    if ph_amendment_needed and "Soil pH Amendment Needed" not in health_statuses:
        health_statuses.append("Soil pH Amendment Needed")
    if ec_intervention_needed and "EC Correction Needed" not in health_statuses:
        health_statuses.append("EC Correction Needed")

    if (fertilizer_needed or ph_amendment_needed or ec_intervention_needed) and (
        "Healthy for Transplanting" in health_statuses
    ):
        health_statuses.remove("Healthy for Transplanting")

    return {
        "Health_Status": health_statuses,
        "NPK": npk_recommendation,
        "PH": ph_recommendation,
        "EC": ec_recommendation,
    }


def phase2_predict(ORP):
    """Run Phase 2 prediction for ORP input."""
    validate_range("ORP_mV", ORP, *ORP_RANGE)

    orp_input = pd.DataFrame([[ORP]], columns=["ORP_mV"])
    orp_prediction = float(orp_model.predict(orp_input)[0])
    flood_liters = max(0.0, min(10000.0, orp_prediction))

    return {
        "Health_Status": get_phase2_health_status(ORP),
        "Phase2_ORP_Flood_Water_Liters": flood_liters,
    }


def print_pretty_result(title, result):
    print(title)
    print("=" * len(title))
    for section, values in result.items():
        print(f"{section}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {values}")
    print()


if __name__ == "__main__":
    phase1_result = phase1_predict(N=40, P=20, K=15, ph=5.5, EC=300)
    phase2_result = phase2_predict(ORP=-100)

    print_pretty_result("Phase 1 Prediction", phase1_result)
    print_pretty_result("Phase 2 Prediction", phase2_result)