from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

def train_classifier(X, y, test_size=0.33, pos_label=1):
    
    # Split data:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Initialize a classifier
    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    
    # Predict probabilities and compute AUC (for binary classifiers):
    prediction_probabilities = clf.predict_proba(X_test)
    if y.nunique(0)[0] == 2:
        auc = roc_auc_score(y_test, prediction_probabilities[:, 1])
        roc = roc_curve(y_test, prediction_probabilities[:, 1], pos_label = pos_label)
    else:
        auc = None
        roc = None
    
    # Predict labels and compute accuracy:
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return {
                "probabilities": prediction_probabilities,
                "predictions": predictions,
                "auc": auc,
                "roc": roc,
                "accuracy": accuracy
            }
