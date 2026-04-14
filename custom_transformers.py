from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class OrdinalEncoderr(BaseEstimator, TransformerMixin):
    def __init__(self, mappings, fillna_map=None):
        self.mappings = mappings
        self.fillna_map = fillna_map or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X.copy()
        for col, mapping in self.mappings.items():
            if col not in Xc.columns:
                continue
            if col in self.fillna_map:
                Xc[col] = Xc[col].fillna(self.fillna_map[col])
            Xc[col] = Xc[col].map(mapping)
            # Bilinmeyen veya mapping dışı kalanları medyan ile doldur
            Xc[col] = Xc[col].fillna(Xc[col].median() if not Xc[col].empty else 0)
        return Xc
