from lib.regime_detection.src.models.hmm import train_hmm, decode_hmm

def train_model(features_df, model_type='hmm'):
    """Generic training interface."""
    if model_type == 'hmm':
        return train_hmm(features_df)
    else:
        raise ValueError(f"Model type {model_type} not supported for training.")

def decode_model(model, state_map, features_df, scaler, model_type='hmm'):
    """Generic decoding interface."""
    if model_type == 'hmm':
        return decode_hmm(model, state_map, features_df, scaler)
    else:
        raise ValueError(f"Model type {model_type} not supported for decoding.")
