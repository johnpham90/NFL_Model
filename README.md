# NFL Prediction Model

## Project Structure
```
NFL_Model/
├── notebooks/           # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb 
│   ├── 03_model_development.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/                # Python source code
│   ├── db_utils.py     # Database connection
│   ├── features.py     # Feature creation
│   ├── model.py        # Model implementation
│   └── evaluation.py   # Model evaluation
│
├── models/            # Saved model files
├── config/           
│   └── db_config.yaml  # Database credentials
│
└── requirements.txt    # Project dependencies
```

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