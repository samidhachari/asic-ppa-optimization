import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# Load training data
try:
    df = pd.read_csv('realistic_asic_data1.csv')
except FileNotFoundError:
    print("Training data file (ealistic_asic_data1.csv) not found. Generating sample data...")
    num_samples = 1000
    data = { 
        'num_gates': np.random.randint(10000, 1000000, num_samples),
        'clock_frequency': np.random.uniform(0.5, 5.0, num_samples),  # GHz
        'input_width': np.random.randint(8, 64, num_samples),
        'pipeline_stages': np.random.randint(1, 10, num_samples),
        'power': [],
        'performance': [],
        'area': []
    } # (Same data generation code as before)
    df = pd.DataFrame(data)
    df.to_csv("asic_design_data.csv", index=False)


X = df[['num_gates', 'clock_frequency', 'input_width', 'pipeline_stages']]
y_power = df['power']
y_performance = df['performance']
y_area = df['area']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_power_train, y_power_test, y_perf_train, y_perf_test, y_area_train, y_area_test = train_test_split(
    X_scaled, y_power, y_performance, y_area, test_size=0.2, random_state=42
)

model_power = RandomForestRegressor(n_estimators=100, random_state=42)
model_performance = RandomForestRegressor(n_estimators=100, random_state=42)
model_area = RandomForestRegressor(n_estimators=100, random_state=42)

model_power.fit(X_train, y_power_train)
model_performance.fit(X_train, y_perf_train)
model_area.fit(X_train, y_area_train)

def evaluate_model(model, X_test, y_test, metric_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{metric_name} - MSE: {mse:.2f}, R-squared: {r2:.2f}")

evaluate_model(model_power, X_test, y_power_test, "Power")
evaluate_model(model_performance, X_test, y_perf_test, "Performance")
evaluate_model(model_area, X_test, y_area_test, "Area")

# Prediction from CSV input
try:
    new_designs = pd.read_csv('predictForThisData.csv')
    new_designs_scaled = scaler.transform(new_designs)

    # predicted_ppa = {
    #     'Power': model_power.predict(new_designs_scaled),
    #     'Performance': model_performance.predict(new_designs_scaled),
    #     'Area': model_area.predict(new_designs_scaled)
    # }

    # for i in range(len(new_designs)):
    #     print(f"Predictions for Design {i+1}:")
    #     for metric, predictions in predicted_ppa.items():
    #         print(f"  {metric}: {predictions[i]:.2f}")

    predicted_power = model_power.predict(new_designs_scaled)
    predicted_performance = model_performance.predict(new_designs_scaled)
    predicted_area = model_area.predict(new_designs_scaled)

    # Combine predictions into a DataFrame
    predicted_ppa = pd.DataFrame({
        "Power": predicted_power,
        "Performance": predicted_performance,
        "Area": predicted_area
    })

    # Concatenate predictions with original input data
    results_df = pd.concat([new_designs, predicted_ppa], axis=1)

    # Save to a new CSV file
    results_df.to_csv("predictForThisData.csv", index=False)
    print("Predictions appended to predictForThisData.csv")

except FileNotFoundError:
    print("Input file (predictForThisData.csv) not found. Please create it.")

#Optimization (using scipy.optimize.minimize)
def power_objective(freq, num_gates, input_width, pipeline_stages, scaler, model):
    test_design = pd.DataFrame({
        'num_gates': [num_gates],
        'clock_frequency': [freq],
        'input_width': [input_width],
        'pipeline_stages': [pipeline_stages]
    })
    test_design_scaled = scaler.transform(test_design)
    power_pred = model.predict(test_design_scaled)[0]
    return power_pred

# Example usage of optimization
num_gates_opt = 500000
input_width_opt = 32
pipeline_stages_opt = 5
initial_freq_guess = 2.5
result = minimize(power_objective, initial_freq_guess, args=(num_gates_opt, input_width_opt, pipeline_stages_opt, scaler, model_power), bounds=[(0.5, 5.0)]) # Frequency bounds

if result.success:
    optimized_freq = result.x[0]
    optimized_power = result.fun
    print(f"Optimized clock frequency for power: {optimized_freq:.2f} GHz, Predicted Power: {optimized_power:.2f}")
else:
    print("Optimization failed.")




