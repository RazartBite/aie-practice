import warnings
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

    preprocessor = build_preprocessor(numeric_features, categorical_features, binary_features)

    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"))
    ])

    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_model("logistic_regression", y_test, baseline_pred, baseline_proba)

    X_train_cat = X_train[numeric_features + categorical_features + binary_features].copy()
    X_val_cat = X_val[numeric_features + categorical_features + binary_features].copy()
    X_test_cat = X_test[numeric_features + categorical_features + binary_features].copy()

    cat_features_idx = [
        X_train_cat.columns.get_loc(col) for col in categorical_features
    ]

    cat_model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_state,
        verbose=False
    )

    cat_model.fit(
        X_train_cat, y_train,
        cat_features=cat_features_idx,
        eval_set=(X_val_cat, y_val),
        use_best_model=True
    )

    cat_pred = cat_model.predict(X_test_cat).astype(int)
    cat_proba = cat_model.predict_proba(X_test_cat)[:, 1]
    cat_metrics = evaluate_model("catboost", y_test, cat_pred, cat_proba)

    metrics = {
        "baseline": baseline_metrics,
        "final_model": cat_metrics
    }

    joblib.dump(baseline, f"{ARTIFACTS_DIR}/baseline_pipeline.joblib")
    cat_model.save_model(f"{ARTIFACTS_DIR}/model.cbm")

    metadata = {
        "model_version": config["service"]["model_version"],
        "features_order": numeric_features + categorical_features + binary_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "target": target
    }

    save_json(metadata, f"{ARTIFACTS_DIR}/metadata.json")
    save_json(metrics, f"{ARTIFACTS_DIR}/metrics.json")

    print("Training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
