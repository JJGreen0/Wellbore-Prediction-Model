import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
import joblib
from tqdm import tqdm  # <— Import tqdm

# ─── USER‐CONFIGURABLE SETTINGS ───────────────────────────────────────────────

# 1) Which column in your DataFrame holds the depth values?
DEPTH_COL = 'Unnamed: 0'

# 2) Which columns should be used as “input features”? 
FEATURE_COLS = ['GR', 'ROP', 'DVER', 'ROPA', 'HKLA', 'TQA', 'SPPA', 'HSX']

# 3) Which column is your target? (Here we predict “future GR” again.)
TARGET_COL = 'GR'

# 4) How far ahead (in the same units as DEPTH_COL) do we look for the target?
LOOKAHEAD = 50  # meters

# 5) Filenames
DATA_FILE         = 'WLC_MUD_LOG_INTERPOLATED.csv'
MODEL_FILE        = 'gr_50m_model.pkl'
PREDICTIONS_FILE  = 'predictions_50m.csv'

# ─── END OF USER‐CONFIGURABLE SETTINGS ────────────────────────────────────────


def prepare_dataset(df: pd.DataFrame, lookahead: float = LOOKAHEAD):
    """
    Builds the feature matrix X and target vector y.
      - df:       Full DataFrame with at least DEPTH_COL, FEATURE_COLS, and TARGET_COL.
      - lookahead: How many depth‐units ahead we want to sample TARGET_COL.
    Returns:
      X: np.ndarray of shape (n_samples, len(FEATURE_COLS))
      y: np.ndarray of shape (n_samples,)
    """
    depths = df[DEPTH_COL].values
    feature_arrays = {col: df[col].values for col in FEATURE_COLS}
    target_array   = df[TARGET_COL].values

    X_rows = []
    y_vals = []

    # Use tqdm to track progress
    for i in tqdm(range(len(df)), desc="Preparing dataset"):
        d = depths[i]

        # 1) If any of the features at row i is NaN, skip
        skip_row = False
        for col in FEATURE_COLS:
            if np.isnan(feature_arrays[col][i]):
                skip_row = True
                break
        if skip_row:
            continue

        # 2) Find the index j such that depth[j] == d + lookahead (or the next‐largest index)
        target_depth = d + lookahead
        j = np.searchsorted(depths, target_depth)
        if j >= len(df):
            # We ran off the end – no future sample available
            break

        # 3) If the actual target (GR at depth j) is NaN, skip
        if np.isnan(target_array[j]):
            continue

        # 4) Append the features at i, and the target at j
        feature_row = [feature_arrays[col][i] for col in FEATURE_COLS]
        X_rows.append(feature_row)
        y_vals.append(target_array[j])

    X = np.array(X_rows)
    y = np.array(y_vals)
    return X, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple = (100, 50),
    activation: str = 'relu',
    solver: str = 'adam',
    alpha: float = 1e-4,
    learning_rate_init: float = 0.001,
    max_iter: int = 500,
    random_state: int = 42
) -> Pipeline:
    """
    Trains a feed‐forward neural network (MLPRegressor) on (X, y) inside a scaling pipeline.
    Returns the fitted Pipeline object.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of length n_samples.
    hidden_layer_sizes : tuple, default=(100, 50)
        Number of neurons in each hidden layer.
    activation : str, default='relu'
        Activation function for the hidden layers.
    solver : str, default='adam'
        The optimizer to use.
    alpha : float, default=1e-4
        L2 penalty (regularization term) parameter.
    learning_rate_init : float, default=0.001
        Initial learning rate for the optimizer.
    max_iter : int, default=500
        Maximum number of iterations (epochs) to train.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline that first scales features (StandardScaler) and then fits an MLPRegressor.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state
        ))
    ])

    pipeline.fit(X, y)
    return pipeline


def generate_predictions(model: Pipeline, df: pd.DataFrame, lookahead: float = LOOKAHEAD):
    """
    Walk through the DataFrame again and make a “real‐time” prediction at each depth i,
    using the same lookahead logic as in prepare_dataset—but here we never “peek” at the actual future target.
    Returns a DataFrame with columns [Depth, Predicted_{TARGET_COL}].
    """
    depths = df[DEPTH_COL].values
    feature_arrays = {col: df[col].values for col in FEATURE_COLS}

    pred_depths = []
    pred_vals   = []

    # Use tqdm to track progress
    for i in tqdm(range(len(df)), desc="Generating predictions"):
        d = depths[i]

        # Skip if any feature is NaN at i
        skip_row = False
        for col in FEATURE_COLS:
            if np.isnan(feature_arrays[col][i]):
                skip_row = True
                break
        if skip_row:
            continue

        target_depth = d + lookahead
        j = np.searchsorted(depths, target_depth)
        if j >= len(df):
            break

        # Build one feature‐vector
        feature_row = np.array([feature_arrays[col][i] for col in FEATURE_COLS]).reshape(1, -1)
        pred = model.predict(feature_row)[0]

        pred_depths.append(d)
        pred_vals.append(pred)

    return pd.DataFrame({
        'Depth': pred_depths,
        f'Predicted_{TARGET_COL}': pred_vals
    })


def main():
    # 1) Read data
    df = pd.read_csv(DATA_FILE)

    # 2) Build (X, y) with LOOKAHEAD logic
    X, y = prepare_dataset(df)
    print(f'Training samples: {X.shape[0]}  |  Number of features: {X.shape[1]}')

    # 3) Train neural‐network model
    model = train_model(
        X, y,
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    )

    # 4) Evaluate on the same (X, y) to report metrics
    y_pred = model.predict(X)

    mae  = mean_absolute_error(y, y_pred)
    mse  = mean_squared_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    r2   = r2_score(y, y_pred)
    evs  = explained_variance_score(y, y_pred)

    print("\n=== Training‐set Metrics ===")
    print(f"MAE:                {mae:.3f}")
    print(f"MSE:                {mse:.3f}")
    print(f"RMSE:               {rmse:.3f}")
    print(f"R²:                 {r2:.3f}")
    print(f"Explained Variance: {evs:.3f}\n")

    # 5) Save the trained pipeline (scaler + MLP) in one file
    joblib.dump(model, MODEL_FILE)
    print(f'Model (pipeline) saved to: {MODEL_FILE}')

    # 6) Re‐run through the file and generate “live” predictions
    pred_df = generate_predictions(model, df)
    pred_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f'Predictions saved to: {PREDICTIONS_FILE}')


if __name__ == '__main__':
    main()
