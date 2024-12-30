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
}

# More Realistic PPA Simulation (Still simplified, but better than before)
for i in range(num_samples):
    gates = data['num_gates'][i]
    freq = data['clock_frequency'][i]
    width = data['input_width'][i]
    stages = data['pipeline_stages'][i]

df = pd.DataFrame(data)
df.to_csv("predictForThisData.csv", index=False)
print(df.head())
print(df.describe()) #Print statistics to verify the data distribution


