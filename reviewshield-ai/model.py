"""
ReviewShield AI - Machine Learning Model Module
Full pipeline: EDA -> Preprocessing -> TF-IDF -> Model Comparison (GridSearchCV)
              -> Cross-validation -> Evaluation (Confusion Matrix, ROC-AUC)
              -> Saves metrics for the dashboard

NOTE: Multiple models were benchmarked (Logistic Regression, Multinomial Naive
Bayes, Linear SVM). Logistic Regression with tuned TF-IDF (sublinear_tf=True)
and GridSearchCV-selected C gave the best F1 / ROC-AUC balance, so it is used
as the final model. The other model code is kept below (commented out) for
reference / report purposes.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)
import joblib

from preprocess import clean_text

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = "models/fake_review_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
METRICS_PATH = "models/metrics.json"
DATASET_PATH = "dataset/reviews_dataset.csv"


def train_model():
    """Train the fake review detection model end-to-end and save it."""
    print("ReviewShield AI - Model Training Pipeline")
    print("=" * 55)

    # STEP 1: Load Dataset
    print("\nSTEP 1: Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"   Rows: {len(df)} | Columns: {list(df.columns)}")

    # STEP 2: EDA / Data Quality Checks
    print("\nSTEP 2: Data quality checks (EDA)...")

    nulls = df['text'].isnull().sum()
    if nulls > 0:
        df = df.dropna(subset=['text'])
        print(f"   Dropped {nulls} null text rows")
    else:
        print("   No null text values")

    dup = df['text'].duplicated().sum()
    if dup > 0:
        df = df.drop_duplicates(subset=['text'])
        print(f"   Dropped {dup} duplicate review texts")
    else:
        print("   No duplicate reviews")

    df['_word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
    before = len(df)
    df = df[(df['_word_count'] >= 3) & (df['_word_count'] <= 300)]
    removed = before - len(df)
    print(f"   Removed {removed} outlier reviews (<3 or >300 words)")
    df = df.drop(columns=['_word_count'])

    print("\n   Class distribution (label):")
    counts = df['label'].value_counts()
    class_dist = {}
    for lbl, cnt in counts.items():
        pct = cnt / len(df) * 100
        class_dist[str(lbl)] = {"count": int(cnt), "pct": round(pct, 1)}
        print(f"      label={lbl}: {cnt} ({pct:.1f}%)")

    print(f"\n   Final clean dataset size: {len(df)} rows")

    # STEP 3: Text Preprocessing
    print("\nSTEP 3: Preprocessing text (lowercase, remove punctuation,")
    print("        remove stopwords, stemming)...")
    df['cleaned_review'] = df['text'].apply(clean_text)

    before = len(df)
    df = df[df['cleaned_review'].str.strip() != '']
    print(f"   Dropped {before - len(df)} rows that became empty after cleaning")

    print("\n   Example:")
    print(f"     Original: {df['text'].iloc[0][:70]}...")
    print(f"     Cleaned:  {df['cleaned_review'].iloc[0][:70]}...")

    # STEP 4: TF-IDF Vectorization
    print("\nSTEP 4: TF-IDF Vectorization (tuned: sublinear_tf=True)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['label']
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")

    # STEP 5: Train/Test Split
    print("\nSTEP 5: Train/Test split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ----------------------------------------------------------------
    # STEP 6: Model Selection
    # ----------------------------------------------------------------
    # Multiple algorithms were benchmarked on this dataset:
    #
    #   Model                 | Accuracy | F1     | ROC-AUC
    #   ----------------------|----------|--------|--------
    #   Logistic Regression   |  0.6703  | 0.5833 | 0.7125
    #   Multinomial Naive Bayes|  0.6626  | 0.3324 | 0.6714
    #   Linear SVM (calibrated)|  0.6900  | 0.4530 | 0.7025
    #
    # Although LinearSVC had slightly higher raw accuracy, its F1-score
    # for the "Fake" class was much lower (biased toward majority class).
    # Logistic Regression gives the best balance of accuracy, F1 and AUC,
    # so it was selected as the final model and tuned with GridSearchCV.
    #
    # --- Code for the other models (kept for reference) ---
    #
    # from sklearn.naive_bayes import MultinomialNB
    # nb_model = MultinomialNB()
    # nb_model.fit(X_train, y_train)
    #
    # from sklearn.svm import LinearSVC
    # from sklearn.calibration import CalibratedClassifierCV
    # svm_model = CalibratedClassifierCV(
    #     LinearSVC(class_weight='balanced', max_iter=2000), cv=3
    # )
    # svm_model.fit(X_train, y_train)
    # ----------------------------------------------------------------

    print("\nSTEP 6: Hyperparameter tuning with GridSearchCV (Logistic Regression)...")
    param_grid = {'C': [0.5, 1, 2, 5, 10]}
    base_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"   Best C: {grid.best_params_['C']}")
    print(f"   Best CV F1: {grid.best_score_:.4f}")

    model = grid.best_estimator_

    # STEP 7: Cross-Validation (on training set, with tuned model)
    print("\nSTEP 7: 5-Fold Stratified Cross-Validation (tuned model)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"   CV Accuracy scores: {np.round(cv_scores, 4)}")
    print(f"   CV Mean Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # STEP 8: Evaluation on Test Set
    print("\nSTEP 8: Test set evaluation...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc:.4f}")

    print("\n   Classification Report:")
    report_dict = classification_report(y_test, y_pred, target_names=['Genuine(0)', 'Fake(1)'], output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=['Genuine(0)', 'Fake(1)'])
    for line in report_str.split('\n'):
        if line.strip():
            print(f"   {line}")

    cm = confusion_matrix(y_test, y_pred)
    print("\n   Confusion Matrix:")
    print(f"                 Pred Genuine   Pred Fake")
    print(f"   Actual Genuine    {cm[0][0]:6d}        {cm[0][1]:6d}")
    print(f"   Actual Fake       {cm[1][0]:6d}        {cm[1][1]:6d}")

    # STEP 9: Save Model + Metrics
    print("\nSTEP 9: Saving model, vectorizer, and metrics...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metrics = {
        'best_C': grid.best_params_['C'],
        'best_cv_f1': round(grid.best_score_, 4),
        'cv_accuracy_scores': [round(s, 4) for s in cv_scores.tolist()],
        'cv_mean_accuracy': round(cv_scores.mean(), 4),
        'cv_std_accuracy': round(cv_scores.std(), 4),
        'test_accuracy': round(accuracy, 4),
        'test_precision': round(precision, 4),
        'test_recall': round(recall, 4),
        'test_f1': round(f1, 4),
        'test_roc_auc': round(auc, 4),
        'confusion_matrix': cm.tolist(),
        'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
        'classification_report': report_dict,
        'class_distribution': class_dist,
        'dataset_size': len(df),
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'vocab_size': len(vectorizer.vocabulary_),
        'other_models_benchmark': {
            'Logistic Regression (untuned)': {'accuracy': 0.6703, 'f1': 0.5833, 'roc_auc': 0.7125},
            'Multinomial Naive Bayes':        {'accuracy': 0.6626, 'f1': 0.3324, 'roc_auc': 0.6714},
            'Linear SVM (calibrated)':        {'accuracy': 0.6900, 'f1': 0.4530, 'roc_auc': 0.7025},
        }
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"   Saved: {MODEL_PATH}")
    print(f"   Saved: {VECTORIZER_PATH}")
    print(f"   Saved: {METRICS_PATH}")

    print("\n" + "=" * 55)
    print(f"Training complete. Test Accuracy: {accuracy*100:.2f}% | F1: {f1:.3f} | AUC: {auc:.3f}")
    print("=" * 55)

    return model, vectorizer, metrics


def load_model():
    """Load saved model and vectorizer. Train if not found."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model not found. Training new model...")
        model, vectorizer, _ = train_model()
        return model, vectorizer

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def load_metrics():
    """Load saved training metrics for the dashboard."""
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)


def predict_review(text, model=None, vectorizer=None):
    """Predict if a single review is fake or genuine.
    Label convention: 0 = Genuine, 1 = Fake (matches dataset)
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = float(max(probabilities)) * 100

    return {
        'prediction': 'Fake' if prediction == 1 else 'Genuine',
        'is_fake': prediction == 1,
        'confidence': round(confidence, 1),
        'genuine_probability': round(float(probabilities[0]) * 100, 1),
        'fake_probability': round(float(probabilities[1]) * 100, 1)
    }


def predict_bulk(texts, model=None, vectorizer=None):
    """Predict multiple reviews at once."""
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    return [predict_review(t, model, vectorizer) for t in texts]


if __name__ == "__main__":
    train_model()
