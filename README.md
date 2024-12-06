# NFL Prediction Model

## Project Structure
```
NFL_MODEL/
├── config/               
│   ├── config.yaml      # Model parameters, features, and settings
│   └── db_config.yaml   # PostgreSQL connection credentials
│
├── notebooks/           
│   ├── 01_data_exploration.ipynb    # Initial data analysis
│   └── 01_data_collection.ipynb     # Data gathering scripts
│
├── src/                 
│   ├── data/           
│   │   ├── external/   # Third-party data (weather, odds)
│   │   ├── processed/  # Cleaned PostgreSQL exports
│   │   └── raw/        # Raw database dumps
│   │
│   ├── features/       
│   │   ├── __init__.py # Package initialization
│   │   └── features.py # Feature engineering functions
│   │
│   ├── models/         
│   │   ├── __init__.py # Package initialization
│   │   └── model.py    # ML model implementation
│   │
│   └── utils/          
│       ├── __init__.py # Package initialization
│       ├── db_utils.py # Database connection handling
│       └── evaluation.py# Model evaluation metrics
│
├── tests/              # Unit tests for code validation
└── requirements.txt    # Project dependencies
```

## Component Usage

### Config Files
- `config.yaml`: Contains model hyperparameters, feature lists, and file paths
- `db_config.yaml`: Stores PostgreSQL credentials securely

### Notebooks
- `01_data_exploration.ipynb`: Initial data analysis and visualization
- `01_data_collection.ipynb`: Scripts for data collection and storage

### Source Code

#### Data Layer
- `external/`: Stores supplementary data like weather conditions and betting odds
- `processed/`: Contains cleaned and transformed data from PostgreSQL
- `raw/`: Stores original database exports

#### Features
- `features.py`: Implements feature engineering for model inputs:
  - Team performance metrics
  - Player statistics
  - Historical matchups
  - Environmental factors

#### Models
- `model.py`: NFL prediction model implementation:
  - Model training
  - Prediction generation
  - Model persistence

#### Utils
- `db_utils.py`: Database interaction functions
- `evaluation.py`: Model performance metrics and validation

## Setup
```bash
pip install -r requirements.txt
```

## Workflow
1. Configure database connection in db_config.yaml
2. Run notebooks for data exploration
3. Use src modules for model development
4. Evaluate results with evaluation.py
