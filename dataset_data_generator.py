from pathlib import Path

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

OUTPUT_FILE = Path(__file__).with_name("KrishiLink_MultiOutput_Training_Data_v4.csv")

# Define dataset split (Total 1500 rows)
num_optimal = 15
num_low_ec = 300
num_trouble = 1185 # Shifted all the extra rows here to create more emergencies

# 1. Trouble Data (Pushed further into extremes to guarantee non-zero outputs)
trouble_df = pd.DataFrame({
    'N': np.random.normal(40, 15, num_trouble).clip(0, 150),
    'P': np.random.normal(20, 10, num_trouble).clip(0, 100),
    'K': np.random.normal(15, 10, num_trouble).clip(0, 100),
    'ph': np.random.normal(5.5, 2.5, num_trouble).clip(3.5, 9.0), # Wider spread for more acid/alkaline
    'EC_uS_cm': np.random.normal(2600, 600, num_trouble).clip(100, 3500), # Higher mean to trigger Phase 1 and Gypsum
    'ORP_mV': np.random.normal(50, 250, num_trouble).clip(-350, 350) # Wider spread to trigger Phase 2 and Lime
})

# 2. Low EC Data (Specifically designed to trigger the Low EC Fertilizer Output)
low_ec_df = pd.DataFrame({
    'N': np.random.normal(20, 10, num_low_ec).clip(0, 100),
    'P': np.random.normal(15, 8, num_low_ec).clip(0, 100),
    'K': np.random.normal(15, 8, num_low_ec).clip(0, 100),
    'ph': np.random.normal(6.5, 0.5, num_low_ec).clip(5.0, 8.0),
    'EC_uS_cm': np.random.normal(150, 80, num_low_ec).clip(0, 299),
    'ORP_mV': np.random.normal(0, 50, num_low_ec).clip(-100, 100)
})

# 3. Optimal Data (Reduced to 15 to prevent the AI from defaulting to 0)
optimal_df = pd.DataFrame({
    'N': np.random.normal(85, 5, num_optimal).clip(75, 150),
    'P': np.random.normal(55, 5, num_optimal).clip(45, 100),
    'K': np.random.normal(45, 5, num_optimal).clip(40, 100),
    'ph': np.random.normal(6.5, 0.3, num_optimal).clip(5.8, 7.2),
    'EC_uS_cm': np.random.normal(1200, 200, num_optimal).clip(500, 1800),
    'ORP_mV': np.random.normal(-20, 40, num_optimal).clip(-100, 100)
})

# Combine and shuffle the dataset
df = pd.concat([trouble_df, low_ec_df, optimal_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the agronomist logic for all 9 outputs
def calculate_phase2_flood_water(orp_mV):
    # Flooding demand should increase smoothly as ORP rises and the soil becomes more oxidized.
    normalized_orp = (orp_mV + 350.0) / 700.0
    normalized_orp = min(max(normalized_orp, 0.0), 1.0)

    # Use a curved response so nearby ORP readings produce distinct water values.
    water_liters = 10000.0 * (normalized_orp ** 1.65)
    return round(water_liters, 2)


def calculate_all_outputs(row):
    # --- 1. HEALTH STATUS ---
    if row['EC_uS_cm'] < 300:
        status = "Severe Depletion (Low EC)"
    elif row['ORP_mV'] < -150:
        status = "Iron Toxicity Risk"
    elif row['EC_uS_cm'] > 2000:
        status = "High Salinity"
    elif row['ph'] > 7.5:
        status = "Alkaline Lockout"
    elif row['N'] < 70 or row['P'] < 40 or row['K'] < 35:
        status = "Nutrient Deficient"
    else:
        status = "Optimal"
        
    # --- 2. FERTILIZER AMOUNTS (kg/acre) ---
    p_def = max(0, 40 - row['P'])
    dap_kg = round(p_def / 0.46, 1) if p_def > 0 else 0.0
    
    n_provided_by_dap = dap_kg * 0.18
    n_def = max(0, 70 - row['N'] - n_provided_by_dap)
    urea_kg = round(n_def / 0.46, 1) if n_def > 0 else 0.0
    
    k_def = max(0, 35 - row['K'])
    mop_kg = round(k_def / 0.60, 1) if k_def > 0 else 0.0

    # --- 3. LOW EC FERTILIZER BOOST ---
    low_ec_boost_kg = 0.0
    if row['EC_uS_cm'] < 300:
        low_ec_boost_kg = round((300 - row['EC_uS_cm']) * 0.25, 1) 

    # --- 4. WATER MANAGEMENT (Liters/acre) ---
    if row['EC_uS_cm'] > 2000:
        phase1_water_L = round((row['EC_uS_cm'] - 2000) * 400, 0)
    else:
        phase1_water_L = 0.0
        
    phase2_water_L = calculate_phase2_flood_water(row['ORP_mV'])
        
    # --- 5. LIME / GYPSUM AMENDMENTS (kg/acre) ---
    lime_kg = 0.0
    gypsum_kg = 0.0
    
    if status == "Iron Toxicity Risk" or row['ph'] < 5.5:
        lime_kg = round(max(50.0, (6.5 - row['ph']) * 150), 1)
        
    if status == "High Salinity" or row['ph'] > 7.5:
        gypsum_kg = round(max(50.0, (row['EC_uS_cm'] - 2000) * 0.05 if row['EC_uS_cm'] > 2000 else 100.0), 1)

    return pd.Series([
        status, 
        urea_kg, dap_kg, mop_kg, low_ec_boost_kg,
        phase1_water_L, phase2_water_L, 
        lime_kg, gypsum_kg
    ])

# Apply calculations to create the output columns
output_cols = [
    'Health_Status', 
    'Urea_kg_per_acre', 'DAP_kg_per_acre', 'MOP_kg_per_acre', 'Low_EC_Fertilizer_Boost_kg',
    'Phase1_EC_Flush_Water_Liters', 'Phase2_ORP_Flood_Water_Liters', 
    'Lime_kg_per_acre', 'Gypsum_kg_per_acre'
]

df[output_cols] = df.apply(calculate_all_outputs, axis=1)


def main():
    # Save the heavily imbalanced (trouble-focused) dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Success! {OUTPUT_FILE.name} has been generated.")
    print("--- Data Distribution Check ---")
    print(f"Optimal Rows: {len(df[df['Health_Status'] == 'Optimal'])}")
    print(f"Rows with non-zero Phase 1 Water (High EC): {len(df[df['Phase1_EC_Flush_Water_Liters'] > 0])}")
    print(f"Rows with non-zero Phase 2 Water (High ORP): {len(df[df['Phase2_ORP_Flood_Water_Liters'] > 0])}")
    print(f"Unique Phase 2 Water values: {df['Phase2_ORP_Flood_Water_Liters'].nunique()}")
    print(f"Rows with non-zero Lime: {len(df[df['Lime_kg_per_acre'] > 0])}")
    print(f"Rows with non-zero Gypsum: {len(df[df['Gypsum_kg_per_acre'] > 0])}")


if __name__ == "__main__":
    main()