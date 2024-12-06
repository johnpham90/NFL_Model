# NFL Prediction Model

## Project Structure
NFL_Model/
├── notebooks/          # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb

├── src/                # Source code
│   ├── db_utils.py     # Database connection/queries
│   ├── features.py     # Feature engineering
│   ├── model.py        # Model implementation
│   └── evaluation.py   # Metrics and validation

├── models/             # Saved model files
├── config/             # Configuration files
└── requirements.txt    # Dependencies

## Setup
```bash
pip install -r requirements.txt
```

## Data Pipeline
1. Query PostgreSQL database
2. Feature engineering
3. Model training
4. Evaluation

## Model Features
- Team performance metrics
- Player statistics
- Historical matchup data
- Environmental factors