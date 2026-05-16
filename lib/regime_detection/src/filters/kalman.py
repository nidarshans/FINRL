import numpy as np
from lib.regime_detection.src.constants import R_BASE, Q_NOISE, GAMMA

class AdaptiveKalmanFilter:
    def __init__(self, R_base=R_BASE, Q=Q_NOISE, gamma=GAMMA):
        self.R_base = R_base
        self.Q      = Q
        self.gamma  = gamma
        self.theta  = 0.0
        self.P      = 1.0

    def filter(self, obs, vol, avg_vol):
        if avg_vol <= 0 or vol <= 0:
            return self.theta, 0.0
        R_t        = self.R_base * (vol / avg_vol) ** self.gamma
        theta_pred = self.theta
        P_pred     = self.P + self.Q
        innovation = obs - theta_pred
        S          = P_pred + R_t
        K          = P_pred / S
        self.theta = theta_pred + K * innovation
        self.P     = (1 - K) * P_pred
        z_score    = innovation / np.sqrt(S)
        return self.theta, z_score


def apply_kalman_to_vf(vf_series, vol_series, avg_vol):
    akf          = AdaptiveKalmanFilter()
    filtered_vf  = []
    innovations  = []
    v_raw_arr    = np.asarray(vf_series, dtype=float)
    vol_arr      = np.asarray(vol_series, dtype=float)
    for i in range(len(v_raw_arr)):
        v_raw    = v_raw_arr[i]
        vol_curr = vol_arr[i]
        if np.isnan(vol_curr) or np.isnan(v_raw):
            filtered_vf.append(0.0)
            innovations.append(0.0)
            continue
        f_val, z_val = akf.filter(v_raw, vol_curr, avg_vol)
        filtered_vf.append(f_val)
        innovations.append(z_val)
    return np.array(filtered_vf), np.array(innovations)
