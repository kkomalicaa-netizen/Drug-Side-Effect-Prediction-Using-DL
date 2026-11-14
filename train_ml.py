import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def main():
    print("Loading data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "dataset/drug_data.csv"))
    
    side_effects = [
        "nausea","headache","dizziness","fatigue","rash",
        "vomiting","insomnia","anxiety","fever","dry_mouth"
    ]
    
    # Create a synthetic binary target: high side effect risk vs low risk
    # Let's say high side effect risk is if they have >= 5 side effects out of 10
    df['target'] = (df[side_effects].sum(axis=1) >= 5).astype(int)
    
    X = df[['description', 'age', 'gender', 'dosage']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age', 'gender', 'dosage']),
            ('text', TfidfVectorizer(max_features=100), 'description')
        ])

    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    metrics = {}
    trained_pipelines = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        
        # calculate roc_auc directly on the valid labels
        roc = 0.0
        if len(np.unique(y_test)) > 1:
            roc = roc_auc_score(y_test, y_proba)
            
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        metrics[name] = {
            'accuracy': float(acc),
            'roc_auc': float(roc),
            'confusion_matrix': cm
        }
        
        trained_pipelines[name] = pipeline
        
    os.makedirs(os.path.join(base_dir, 'model'), exist_ok=True)
    
    with open(os.path.join(base_dir, 'model/ml_models.pkl'), 'wb') as f:
        pickle.dump(trained_pipelines, f)
        
    with open(os.path.join(base_dir, 'model/ml_metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    print("All ML models trained and saved successfully.")

if __name__ == "__main__":
    main()
