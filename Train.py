import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

LOOKAHEAD = 50  # meters
DATA_FILE = 'WLC_MUD_LOG_INTERPOLATED.csv'
MODEL_FILE = 'gr_50m_model.pkl'
PREDICTIONS_FILE = 'predictions_50m.csv'


def prepare_dataset(df, lookahead=LOOKAHEAD):
    depths = df['Unnamed: 0'].values
    gr = df['GR'].values
    rop = df['ROP'].values

    features = []
    targets = []

    # Precompute searchsorted on depths
    for i in range(len(df)):
        d = depths[i]
        cur_gr = gr[i]
        cur_rop = rop[i]
        if np.isnan(cur_gr) or np.isnan(cur_rop):
            continue
        target_depth = d + lookahead
        j = np.searchsorted(depths, target_depth)
        if j >= len(df):
            break
        target_gr = gr[j]
        if np.isnan(target_gr):
            continue
        features.append([d, cur_gr, cur_rop])
        targets.append(target_gr)

    return np.array(features), np.array(targets)


def train_model(features, targets):
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(features, targets)
    return model


def generate_predictions(model, df, lookahead=LOOKAHEAD):
    depths = df['Unnamed: 0'].values
    gr = df['GR'].values
    rop = df['ROP'].values

    pred_depths = []
    preds = []
    for i in range(len(df)):
        d = depths[i]
        if np.isnan(gr[i]) or np.isnan(rop[i]):
            continue
        target_depth = d + lookahead
        j = np.searchsorted(depths, target_depth)
        if j >= len(df):
            break
        # Predict using model
        pred = model.predict([[d, gr[i], rop[i]]])[0]
        pred_depths.append(d)
        preds.append(pred)
    return pd.DataFrame({'Depth': pred_depths, 'Predicted_GR': preds})


def main():
    df = pd.read_csv(DATA_FILE)
    features, targets = prepare_dataset(df)
    print(f'Training samples: {len(features)}')
    model = train_model(features, targets)
    joblib.dump(model, MODEL_FILE)
    print(f'Model saved to {MODEL_FILE}')
    pred_df = generate_predictions(model, df)
    pred_df.to_csv(PREDICTIONS_FILE, index=False)
    print(f'Predictions saved to {PREDICTIONS_FILE}')


if __name__ == '__main__':
    main()
