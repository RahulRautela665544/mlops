"Predict one delivery, from the command line."

import pandas as pd
from pathlib import Path
from delivery import load_model

def main():
    # Load the model from the current directory (work/)
    model = load_model(Path(__file__).parent / "model.joblib")

    # Build a one-row dataframe for the prediction
    order_data = pd.DataFrame([{
        "distance_km": 7.0, 
        "prep_time_min": 25, 
        "traffic_level": 3, 
        "rain": 0
    }])

    # Predict and print
    minutes = float(model.predict(order_data)[0])
    print(f"PREDICTION: {minutes:.1f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
