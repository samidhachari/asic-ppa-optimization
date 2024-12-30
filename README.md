## Project Overview

Predicting PPA is crucial in ASIC design. Traditionally, this involves time-consuming simulations. This project explores using machine learning models to predict PPA based on design parameters, allowing for rapid evaluation of different design configurations.

## Features

*   Predicts Power, Performance, and Area based on design parameters.
*   Uses Random Forest Regression for prediction.
*   Includes data scaling for improved model performance.
*   Provides an example of clock frequency optimization using `scipy.optimize.minimize`.
*   Appends predictions to the input CSV file.
*   Handles missing training data by generating synthetic data.

## Files

*   `asic_optimizer.py`: The main Python script containing the code for data loading, model training, prediction, and optimization.
*   `realistic_asic_data1.csv`: Example training data (can be generated if not present).
*   `predictForThisData.csv`: Example input file for predictions.
*   `README.md`: This file.

## Getting Started

1.  Clone the repository:

    ```bash
    git clone [https://github.com/YourUsername/asic-ppa-optimization.git](https://github.com/YourUsername/asic-ppa-optimization.git)
    ```

2.  Install required libraries:

    ```bash
    pip install pandas numpy scikit-learn scipy
    ```

3.  Prepare your prediction input data: Create a CSV file named `predictForThisData.csv` with the following columns: `num_gates`, `clock_frequency`, `input_width`, `pipeline_stages`.

4.  Run the script:

    ```bash
    python asic_optimizer.py
    ```

5. The predicted PPA values will be appended to your `predictForThisData.csv` file.

## Usage

The script will first train the models (or load the training data if it exists). Then, it will read data from `predictForThisData.csv`, make predictions, and save them back to the same file. The optimization example will also run, printing the optimized clock frequency and power.

## Example Input (predictForThisData.csv)

```csv
num_gates,clock_frequency,input_width,pipeline_stages
100000,1.0,16,2
200000,2.0,32,4
