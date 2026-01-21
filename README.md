# Exoplanet Detection with Neural Networks

A machine learning project to classify Kepler objects as planets or false positives.

---

## Goal

Build a neural network that predicts whether a Kepler Object of Interest (KOI) is a real exoplanet or a false positive, using transit and stellar features.

---

## Data

**Source:** NASA Exoplanet Archive — Kepler Cumulative Table

**Training set:** 7,326 labeled objects
- Confirmed planets: 2,744
- False positives: 4,582

**Features (9):**

| Feature | Description |
|---------|-------------|
| koi_period | Orbital period (days) |
| koi_depth | Transit depth (ppm) |
| koi_duration | Transit duration (hours) |
| koi_prad | Planet radius (Earth radii) |
| koi_teq | Equilibrium temperature (K) |
| koi_insol | Insolation flux (Earth flux) |
| koi_steff | Star effective temperature (K) |
| koi_srad | Star radius (Solar radii) |
| koi_model_snr | Signal-to-noise ratio |

---

## Model

**Architecture:** Fully connected neural network

```
Input (9 features)
    ↓
Linear(128) → BatchNorm → ReLU → Dropout(0.4)
    ↓
Linear(64) → BatchNorm → ReLU → Dropout(0.4)
    ↓
Linear(32) → BatchNorm → ReLU
    ↓
Linear(1) → Sigmoid
    ↓
Output (probability)
```

**Classification threshold:** 0.7

---

## Results

### Model Evolution

| Version | Features | Architecture | Accuracy | F1 Score |
|---------|----------|--------------|----------|----------|
| V1 | 3 | 3→32→16→1 | 80.76% | 75.31% |
| V2 | 3 | 3→64→32→16→1 + dropout | 81.65% | 76.75% |
| V3 | 9 | 9→128→64→32→1 + batchnorm | **85.81%** | **82.95%** |

### Final Performance (V3)

| Metric | Score |
|--------|-------|
| Accuracy | 85.81% |
| Precision | 75.41% |
| Recall | 92.17% |
| F1 Score | 82.95% |

### Validation Against NASA Scores

| Metric | Value |
|--------|-------|
| Agreement rate | 73.5% |
| Correlation | 0.40 |
| Strong agreement candidates | 466 |

---

## Predictions

**Unconfirmed candidates evaluated:** 1,875

| Prediction | Count |
|------------|-------|
| Likely planet | 1,068 |
| Likely false positive | 807 |

---

## Project Structure

```
exoplanet-detection/
├── data/
│   ├── raw/                    # Downloaded light curves
│   └── processed/              # Train/test arrays
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
├── models/
│   ├── exoplanet_nn_v3.pt      # Trained model
│   └── scaler.pkl              # Feature scaler
├── notebooks/
│   └── 01_data_exploration.ipynb
├── results/
│   ├── candidate_predictions.csv
│   ├── strong_planet_candidates.csv
│   └── validation_vs_nasa.csv
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/MikeMurgia/AIExoplanetFinder.git
cd AIExoplanetFinder
pip install -r requirements.txt
```

**Requirements:**
```
numpy
pandas
matplotlib
scikit-learn
torch
lightkurve
```

---

## Usage

### Load and Predict

```python
import torch
import pickle
import numpy as np

# Load model
model = ExoplanetNNv3(input_size=9)
model.load_state_dict(torch.load("models/exoplanet_nn_v3.pt"))
model.eval()

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Predict
features = [period, depth, duration, prad, teq, insol, steff, srad, snr]
X = scaler.transform([features])
X = torch.tensor(X, dtype=torch.float32)

with torch.no_grad():
    prob = model(X).item()

prediction = "Planet" if prob > 0.7 else "False Positive"
```

---

## Key Findings

1. **More features improve performance** — jumping from 3 to 9 features increased accuracy by 5%

2. **Threshold tuning matters** — 0.7 threshold balanced precision/recall better than default 0.5

3. **Model agrees with NASA 73.5% of the time** — disagreements often involve extreme radius values

4. **466 strong candidates** — both our model and NASA agree these are likely planets

---

## Future Improvements

- [ ] Use raw light curves with phase folding
- [ ] Implement CNN for time-series classification
- [ ] Add cross-validation
- [ ] Try ensemble methods (Random Forest + NN)
- [ ] Handle class imbalance with weighted loss

---

## References

- NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu
- Kepler Mission: https://www.nasa.gov/mission_pages/kepler
- Lightkurve: https://docs.lightkurve.org

---
