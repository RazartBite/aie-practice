import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from catboost import CatBoostClassifier

from src.utils import load_yaml, save_json, ensure_dir


def build_preprocessor(numeric_features, categorical_features, binary_features):
    """Создание пайплайна предобработки"""
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    bin_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_features),
        ('cat', cat_pipe, categorical_features),
        ('bin', bin_pipe, binary_features)
    ])
    
    return preprocessor


def evaluate_model(model_name, y_true, y_pred, y_proba):
    """Оценка качества модели"""
    metrics = {
        'model': model_name,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'f1': round(f1_score(y_true, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_true, y_proba), 4)
    }
    return metrics


def main():
    # Загрузка конфигурации
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
    config = load_yaml(config_path)
    
    random_state = config['random_state']
    data_path = Path(__file__).parent.parent / config['data']['raw_csv_path']
    target = config['features']['target']
    drop_columns = config['features']['drop_columns']
    numeric_features = config['features']['numeric']
    categorical_features = config['features']['categorical']
    binary_features = config['features']['binary']
    test_size = config['train']['test_size']
    val_size = config['train']['val_size']
    
    # Путь для сохранения артефактов
    ARTIFACTS_DIR = Path(__file__).parent.parent / 'artifacts'
    ensure_dir(ARTIFACTS_DIR)
    
    # 1. Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv(data_path)
    
    # 2. Очистка
    df_clean = df.drop(columns=drop_columns).drop_duplicates().copy()
    
    all_features = numeric_features + categorical_features + binary_features
    X = df_clean[all_features]
    y = df_clean[target]
    
    # 3. Разделение данных
    print("Разделение данных...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    val_ratio_from_train = val_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_ratio_from_train,
        stratify=y_train_full,
        random_state=random_state
    )

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # 4. Baseline: Logistic Regression
    print("\nОбучение Logistic Regression...")
    preprocessor = build_preprocessor(numeric_features, categorical_features, binary_features)

    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"))
    ])

    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_model("logistic_regression", y_test, baseline_pred, baseline_proba)
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['f1']:.4f}, ROC-AUC: {baseline_metrics['roc_auc']:.4f}")

    # 5. CatBoost Base (добавлено)
    print("\nОбучение CatBoost Base...")
    X_train_cat = X_train[numeric_features + categorical_features + binary_features].copy()
    X_val_cat = X_val[numeric_features + categorical_features + binary_features].copy()
    X_test_cat = X_test[numeric_features + categorical_features + binary_features].copy()

    cat_features_idx = [
        X_train_cat.columns.get_loc(col) for col in categorical_features
    ]

    catboost_base = CatBoostClassifier(
        iterations=150,          # базовые параметры
        depth=5,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False
    )

    catboost_base.fit(
        X_train_cat, y_train,
        cat_features=cat_features_idx,
        eval_set=(X_val_cat, y_val),
        use_best_model=True
    )

    cat_base_pred = catboost_base.predict(X_test_cat).astype(int)
    cat_base_proba = catboost_base.predict_proba(X_test_cat)[:, 1]
    cat_base_metrics = evaluate_model("catboost_base", y_test, cat_base_pred, cat_base_proba)
    print(f"  Accuracy: {cat_base_metrics['accuracy']:.4f}, F1: {cat_base_metrics['f1']:.4f}, ROC-AUC: {cat_base_metrics['roc_auc']:.4f}")

    # 6. CatBoost Final (существующий код, переименован)
    print("\nОбучение CatBoost Final...")
    catboost_final = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False
    )

    catboost_final.fit(
        X_train_cat, y_train,
        cat_features=cat_features_idx,
        eval_set=(X_val_cat, y_val),
        use_best_model=True
    )

    cat_final_pred = catboost_final.predict(X_test_cat).astype(int)
    cat_final_proba = catboost_final.predict_proba(X_test_cat)[:, 1]
    cat_final_metrics = evaluate_model("catboost_final", y_test, cat_final_pred, cat_final_proba)
    print(f"  Accuracy: {cat_final_metrics['accuracy']:.4f}, F1: {cat_final_metrics['f1']:.4f}, ROC-AUC: {cat_final_metrics['roc_auc']:.4f}")

    # 7. Сохранение артефактов
    print("\nСохранение артефактов...")

    joblib.dump(baseline, f"{ARTIFACTS_DIR}/baseline_pipeline.joblib")
    catboost_final.save_model(f"{ARTIFACTS_DIR}/model.cbm")

    metadata = {
        "model_version": config["service"]["model_version"],
        "features_order": numeric_features + categorical_features + binary_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "target": target
    }

    save_json(metadata, f"{ARTIFACTS_DIR}/metadata.json")

    # metrics.json теперь содержит все 3 модели
    metrics = {
        "baseline": baseline_metrics,
        "model_a": cat_base_metrics,      # добавлено
        "model_b": cat_final_metrics,     # было final_model
        "best_params": {
            "loss_function": "Logloss",
            "random_seed": random_state,
            "verbose": False,
            "eval_metric": "AUC",
            "depth": 6,
            "iterations": 300,
            "learning_rate": 0.05
        }
    }
    save_json(metrics, f"{ARTIFACTS_DIR}/metrics.json")

    print("\n✅ Training complete.")
    print("\nСравнение моделей:")
    print("-" * 60)
    print(f"{'Модель':<20} {'Accuracy':<10} {'F1-score':<10} {'ROC-AUC':<10}")
    print("-" * 60)
    print(f"{'Logistic Regression':<20} {baseline_metrics['accuracy']:<10.4f} {baseline_metrics['f1']:<10.4f} {baseline_metrics['roc_auc']:<10.4f}")
    print(f"{'CatBoost Base':<20} {cat_base_metrics['accuracy']:<10.4f} {cat_base_metrics['f1']:<10.4f} {cat_base_metrics['roc_auc']:<10.4f}")
    print(f"{'CatBoost Final':<20} {cat_final_metrics['accuracy']:<10.4f} {cat_final_metrics['f1']:<10.4f} {cat_final_metrics['roc_auc']:<10.4f}")
    print("-" * 60)


if __name__ == "__main__":
    main()
