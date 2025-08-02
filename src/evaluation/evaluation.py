from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def analyze_errors(y_true, y_pred, data):
    '''Analyze prediction errors'''
    errors = y_true != y_pred
    error_analysis = {}
    # Add error analysis logic
    return error_analysis
