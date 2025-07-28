# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/muyi/shell_competition/type2/random_forest.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-07-21 16:22:54 UTC (1753114974)

import pandas as pd
import time
from sklearn.model_selection import cross_val_score
from xgboost_multioutput import XGBoostMultiOutputRegressor

class RandomForestMultiOutputRegressor:

    def __init__(self, n_estimators=100, random_state=None, **kwargs):
        self.model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, **kwargs))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        X: pandas DataFrame of shape (n_samples, 55)
        y: pandas DataFrame of shape (n_samples, 10)
        """
        assert X.shape[1] == 55, f'Expected 55 features, got {X.shape[1]}'
        assert y.shape[1] == 10, f'Expected 10 output columns, got {y.shape[1]}'
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        X: pandas DataFrame of shape (n_samples, 55)
        Returns: pandas DataFrame of shape (n_samples, 10)
        """
        assert X.shape[1] == 55, f'Expected 55 features, got {X.shape[1]}'
        preds = self.model.predict(X)
        return pd.DataFrame(preds, index=X.index)
pass
pass
pass

def grid_search_best_random_forests(X: pd.DataFrame, y: pd.DataFrame, n_estimators_list, random_state_list, top_k=10, scoring=None, cv=None, **model_kwargs):
    """
    Iterates over combinations of n_estimators and random_state, fits models, and returns the top_k best models.
    Supports cross-validation if cv is specified.
    
    Parameters:
        X: DataFrame of shape (n_samples, 55)
        y: DataFrame of shape (n_samples, 10)
        n_estimators_list: list of int
        random_state_list: list of int
        top_k: int, number of best models to return
        scoring: str or callable, scoring metric for cross_val_score or model.score
        cv: int, cross-validation folds (if None, no CV, use model.score on full data)
        model_kwargs: additional kwargs for RandomForestRegressor
    Returns:
        List of tuples: (mean_score, model, n_estimators, random_state)
    """
    results = []
    total_models = len(n_estimators_list) * len(random_state_list)
    model_count = 0
    start_time = time.time()
    for n_estimators in n_estimators_list:
        for random_state in random_state_list:
            model_count += 1
            print(f'\n[{model_count}/{total_models}] Fitting model with n_estimators={n_estimators}, random_state={random_state}...')
            model_start = time.time()
            model = XGBoostMultiOutputRegressor(n_estimators=n_estimators, random_state=random_state, **model_kwargs)
            if cv is not None:
                scores = cross_val_score(model.model, X, y, scoring=scoring, cv=cv)
                mean_score = scores.mean()
            else:
                model.fit(X, y)
                if scoring is not None and callable(scoring):
                    mean_score = scoring(model, X, y)
                else:
                    mean_score = model.model.score(X, y)
            model_end = time.time()
            elapsed = model_end - model_start
            total_elapsed = model_end - start_time
            avg_per_model = total_elapsed / model_count
            est_remaining = avg_per_model * (total_models - model_count)
            print(f'    Model fit time: {elapsed:.2f} seconds.')
            print(f'    Estimated time remaining: {est_remaining / 60:.2f} minutes.')
            results.append((mean_score, model, n_estimators, random_state))
            
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]