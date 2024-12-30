import pandas as pd
import numpy as np
import random

np.random.seed(42)  # for reproducibility
num_samples = 1000

# Realistic Ranges (Adjust as needed based on your target designs)
num_gates_range = (10000, 5000000)  # Example: 10k to 5M gates
clock_frequency_range = (0.5, 5.0)  # Example: 0.5 GHz to 5 GHz
input_width_range = (8, 128)      # Example: 8 bits to 128 bits
pipeline_stages_range = (1, 20)    # Example: 1 to 20 pipeline stages

data = {
    'num_gates': np.random.randint(*num_gates_range, num_samples),
    'clock_frequency': np.random.uniform(*clock_frequency_range, num_samples),
    'input_width': np.random.randint(*input_width_range, num_samples),
    'pipeline_stages': np.random.randint(*pipeline_stages_range, num_samples),
    'power': [],
    'performance': [],
    'area': []
}

# More Realistic PPA Simulation (Still simplified, but better than before)
for i in range(num_samples):
    gates = data['num_gates'][i]
    freq = data['clock_frequency'][i]
    width = data['input_width'][i]
    stages = data['pipeline_stages'][i]

    # Power (Dynamic power is dominant, proportional to f*C*V^2, simplified)
    power = (0.000001 * gates * freq**2) + (0.001 * width * freq) + (0.0001 * stages) + np.random.normal(0, 0.5) #Adding noise

    # Performance (Simplified, but considers pipelining)
    performance = (freq * stages) / (width**0.5) + np.random.normal(0, 0.2)

    # Area (Related to gate count and input width)
    area = (0.0000001 * gates) + (0.001 * width**2) + np.random.normal(0, 0.1)

    data['power'].append(max(0,power)) #Power should not be negative
    data['performance'].append(performance)
    data['area'].append(max(0,area)) #Area should not be negative

df = pd.DataFrame(data)
df.to_csv("realistic_asic_data.csv", index=False)
print(df.head())
print(df.describe()) #Print statistics to verify the data distribution