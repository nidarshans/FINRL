import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
from lib.regime_detection.src.constants import FEATURES, HMM_ITER

def train_hmm(features_df):
    features = features_df[FEATURES].dropna()
    if len(features) < 30:
        return None, None, None, None

    scaler      = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    model = GMMHMM(
        n_components=3, n_mix=3,
        covariance_type="diag",
        n_iter=HMM_ITER, random_state=42
    )
    model.startprob_ = np.array([0.33, 0.33, 0.34])
    model.transmat_  = np.array([
        [0.95, 0.04, 0.01],
        [0.05, 0.90, 0.05],
        [0.01, 0.04, 0.95],
    ])
    try:
        model.fit(scaled_data)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  [HMM] training failed: {e}")
        return None, None, None, None

    state_means = [
        np.average(model.means_[i], axis=0, weights=model.weights_[i])[0]
        for i in range(3)
    ]
    order     = np.argsort(state_means)
    state_map = {int(order[0]): "Bear", int(order[1]): "Stagnant", int(order[2]): "Bull"}
    return model, state_map, features.index, scaler


def decode_hmm(model, state_map, features_df, scaler):
    features = features_df[FEATURES].dropna()
    if model is None or len(features) == 0 or scaler is None:
        return pd.DataFrame()
    scaled = scaler.transform(features)
    try:
        hidden_states = model.predict(scaled)
        probs         = model.predict_proba(scaled)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  [HMM] decode failed: {e}")
        return pd.DataFrame()

    bull_idx = [k for k, v in state_map.items() if v == "Bull"][0]
    bear_idx = [k for k, v in state_map.items() if v == "Bear"][0]

    out             = pd.DataFrame(index=features.index)
    out['Regime']   = [state_map[s] for s in hidden_states]
    out['P_Bull']   = probs[:, bull_idx]
    out['P_Bear']   = probs[:, bear_idx]
    out['Rank_Score'] = out['P_Bull'] - out['P_Bear'] + (scaled[:, 1] * 0.1)
    return out
