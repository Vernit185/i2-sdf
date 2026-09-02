import numpy as np
import torch
import warnings

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class PhysicsFeatureExtractor:
    def __init__(self, v_up, h_floor):
        self.v_up = v_up.cpu().numpy()
        self.h_floor = float(h_floor.cpu().item())

    def extract_features(self, surface_points, column_sdfs, normals=None):
        """
        Extracts geometric and physical features from reconstructed SDF columns.
        
        Args:
            surface_points: (N, 3) numpy array of candidate points.
            column_sdfs: (N, K) numpy array of vertical SDF probes.
            normals: Optional (N, 3) numpy array of surface normals.
            
        Returns:
            features: (N, F) numpy array of features.
        """
        N, K = column_sdfs.shape
        features = []
        
        # 1. Height above floor
        heights = np.dot(surface_points, self.v_up)
        h_diff = np.clip(heights - self.h_floor, 0, None)
        
        # 2. SDF statistics along the column
        min_sdf = np.min(column_sdfs, axis=1)
        mean_sdf = np.mean(column_sdfs, axis=1)
        var_sdf = np.var(column_sdfs, axis=1)
        
        for i in range(N):
            feat = [h_diff[i], min_sdf[i], mean_sdf[i], var_sdf[i]]
            if normals is not None:
                # 3. Normal alignment with upward axis
                vertical_alignment = np.dot(normals[i], self.v_up)
                feat.append(vertical_alignment)
            features.append(feat)
            
        return np.array(features)

class PhysicsRandomForestAnalyzer:
    def __init__(self, mode='classification'):
        self.mode = mode
        self.model = None
        if not SKLEARN_AVAILABLE:
            warnings.warn("scikit-learn is not available. Random Forest analyzer is disabled.")
            
    def fit_and_evaluate(self, X, y=None):
        """
        Trains and evaluates the Random Forest if ground truth labels 'y' are provided.
        DOES NOT fabricate labels if 'y' is missing.
        """
        if not SKLEARN_AVAILABLE:
            print("[INFO] scikit-learn is required for Random Forest analysis.")
            return None
            
        if y is None or len(np.unique(y)) < 2:
            print("[INFO] Labeled dataset is unavailable for Random Forest supervised evaluation.")
            print("[INFO] No fabricated labels were created. Supervised ML component skipped.")
            return None
            
        if self.mode == 'classification':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        
        metrics = {}
        if self.mode == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, preds)
            metrics['report'] = classification_report(y_test, preds, output_dict=True)
            print(f"[INFO] Random Forest Classification Accuracy: {metrics['accuracy']:.4f}")
        
        if hasattr(self.model, 'feature_importances_'):
            metrics['feature_importances'] = self.model.feature_importances_
            print(f"[INFO] Random Forest Feature Importances: {metrics['feature_importances']}")
            
        return metrics
