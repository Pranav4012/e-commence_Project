# Delivery Intelligence

A modern React + TypeScript dashboard with a Python FastAPI data service for the project **Generalized E-Commerce Delivery Delay Prediction Using Multi-Factor Data**.

## Start the complete application

Requirements:
- Node.js 20 or newer
- Python 3.10 or newer

In the project folder:

```bash
npm install
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm run dev
```

**Windows CMD**
```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
npm run dev
```

**macOS / Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
npm run dev
```

Open the local Vite address printed in the terminal (normally `http://localhost:5173`). `npm run dev` starts both the React frontend and FastAPI backend. Keep the terminal running.

## Current functionality

- React dashboard with responsive navigation and interactive chart components.
- CSV upload to the Python API.
- Actual row/column counts, missingness, exact duplicate count, memory footprint, column types, unique counts and preview.
- Date-like column detection and monthly record counts.
- Numeric descriptive summaries.
- User-configured classification training with explicit target and feature selection.
- Holdout accuracy, macro precision/recall/F1, per-class results, confusion matrix and sample predictions.
- Single-record predictions with explicitly uncalibrated class probabilities from the in-memory trained model.
- Searchable package tracking from uploaded order IDs and CSV statuses, with monthly delivery statistics and status charts.
- Historical-cohort arrival estimates and possible recorded delay signals for incomplete orders; source CSV values are not overwritten.
- Interactive analytics with time-window and chart controls, delivery-status composition, and a clickable missingness chart.
- Dark and light appearance modes, with the selected theme saved in the browser.
- Random-forest global feature importance and local single-feature baseline-sensitivity explanations.
- A held-out one-feature-at-a-time missing-input robustness experiment.

## Important research integrity

The master dataset `india_ecommerce_master_650k.csv` has not been included. Upload your actual CSV through **Import dataset**. The Order prediction page trains a baseline random forest classifier from the target and feature columns you select. It can show real results only after you upload and configure a labeled dataset.

Before interpreting results:
1. Verify data provenance and whether records are original or augmented.
2. Select a verified delivery outcome as the target and only checkout-time fields as features.
3. Exclude post-order leakage fields such as actual shipping/delivery dates, delivery status, actual days and profit if unavailable at checkout.
4. Select a verified order-time field for chronological holdout evaluation when available. Without one, the baseline uses a stratified random split.
5. Build seller/route historical aggregates using earlier orders only before including them as features.

The baseline classifier and its test metrics are not a completed research evaluation. The model exists only in API memory and is cleared when a new dataset is uploaded or the API restarts. Global importance uses Random Forest impurity importance; local explanations compare one input at a time with a training-set median or mode. These are sensitivity aids, not SHAP values, causal effects, or calibrated probabilities. Robustness results measure the effect of masking individual inputs on a fixed sample of held-out rows; they are not a guarantee of performance under real-world distribution shift. Regression is not implemented.

Package tracking reads the latest uploaded CSV snapshot (restored locally after an API restart). It is not connected to carrier scan feeds. Estimated arrival probabilities are empirical shares among completed comparable orders in that file; they are not calibrated forecasts. Remaining days are shown only while the historical ETA is still ahead; past ETAs are called out as overdue without inventing a new carrier estimate. The dataset has no verified event-reason field, so weather flags and high operational values are shown only as possible signals—not confirmed causes such as a landslide. Conflicting incomplete delivery dates are ignored.

## Structure

```text
delivery-intelligence/
├── backend/
│   └── main.py
├── src/
│   ├── App.tsx
│   ├── main.tsx
│   └── styles.css
├── data/uploads/
├── models/
├── package.json
├── requirements.txt
└── vite.config.ts
```

Uploaded CSVs are stored locally in `data/uploads/`. Do not commit private or sensitive datasets.
