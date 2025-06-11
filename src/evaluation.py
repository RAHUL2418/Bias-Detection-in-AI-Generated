from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred):
    """Print classification metrics."""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
