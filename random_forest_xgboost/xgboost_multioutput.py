# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /home/muyi/shell_competition/type2/xgboost_multioutput.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-07-22 02:51:48 UTC (1753152708)

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

class XGBoostMultiOutputRegressor:

    def __init__(self, n_estimators=100, random_state=None, tree_method='hist', device='cuda', **kwargs):
        """
        n_estimators: Number of boosting rounds.
        random_state: Random seed.
        tree_method: 'hist' enables GPU acceleration with device='cuda' in XGBoost >=2.0.
        device: 'cuda' to use GPU (XGBoost >=2.0)
        kwargs: Additional parameters for XGBRegressor.
        """
        self.model = MultiOutputRegressor(XGBRegressor(n_estimators=n_estimators, random_state=random_state, tree_method=tree_method, device=device, **kwargs))

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = self.model.predict(X)
        return pd.DataFrame(preds, index=X.index)