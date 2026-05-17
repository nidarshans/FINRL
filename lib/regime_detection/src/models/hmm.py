import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.preprocessing import StandardScaler
from lib.regime_detection.src.constants import FEATURES, HMM_ITER

def _make_transmat(corr_stress: float = 0.0) -> np.ndarray:
    """
    Return a 3x3 transition matrix biased by cross-sector correlation stress.
    corr_stress in [0, 1]: 0 = calm, 1 = maximum systemic risk.
    """
    stress = np.clip(corr_stress, 0.0, 1.0)
    
    # Stay probabilities scale down under stress for calm regimes,
    # and stay in Bear slightly longer
    p_stay_bull = max(0.80, 0.95 - 0.15 * stress)
    p_stay_stag = max(0.75, 0.90 - 0.15 * stress)
    p_stay_bear = min(0.98, 0.95 + 0.03 * stress)
    
    p_bear_to_bull = 0.01
    p_bear_to_stag = 1.0 - p_stay_bear - p_bear_to_bull
    
    p_stag_to_bear = 0.05 + 0.10 * stress
    p_stag_to_bull = 1.0 - p_stay_stag - p_stag_to_bear
    if p_stag_to_bull < 0.01:
        p_stag_to_bull = 0.01
        p_stay_stag = 1.0 - p_stag_to_bear - p_stag_to_bull
        
    p_bull_to_bear = 0.01 + 0.09 * stress
    p_bull_to_stag = 1.0 - p_stay_bull - p_bull_to_bear
    if p_bull_to_stag < 0.01:
        p_bull_to_stag = 0.01
        p_stay_bull = 1.0 - p_bull_to_bear - p_bull_to_stag
        
    transmat = np.array([
        [p_stay_bear, p_bear_to_stag, p_bear_to_bull],
        [p_stag_to_bear, p_stay_stag, p_stag_to_bull],
        [p_bull_to_bear, p_bull_to_stag, p_stay_bull]
    ])
    return transmat


def train_hmm(features_df):
    features = features_df[FEATURES].dropna()
    if len(features) < 30:
        return None, None, None, None

    scaler      = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    model = GMMHMM(
        n_components=3, n_mix=3,
        covariance_type="diag",
        n_iter=HMM_ITER, random_state=42,
        init_params="wmc"
    )
    
    corr_stress = 0.0
    if 'Absorption_Ratio' in features.columns:
        corr_stress = float(features['Absorption_Ratio'].mean())

    model.startprob_ = np.array([0.33, 0.33, 0.34])
    model.transmat_  = _make_transmat(corr_stress)
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
    #In the future, we would want an ML algorithm to output rank score instead of hardcoding it
    if 'Absorption_Ratio' in FEATURES:
        from lib.regime_detection.src.constants import AR_SCORE_WEIGHT
        ar_idx = FEATURES.index('Absorption_Ratio')
        out['Rank_Score'] = out['P_Bull'] - out['P_Bear'] - (scaled[:, ar_idx] * AR_SCORE_WEIGHT)
    else:
        out['Rank_Score'] = out['P_Bull'] - out['P_Bear'] + (scaled[:, 1] * 0.1)
    return out
