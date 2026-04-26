import re
from typing import Optional

import numpy as np
import pandas as pd

from simple_quant import config

def prepare_modeling_data(
    df_input: pd.DataFrame,
    ticker: str,
    windows: list[int] = config.DEFAULT_WINDOWS,
    macro_cols: list[str] = config.DEFAULT_MACRO_COLS,
) -> pd.DataFrame:
    df = df_input.sort_values("data").reset_index(drop=True).copy()

    target_price_col = config.price_column_name(ticker)
    if target_price_col not in df.columns:
        raise ValueError(f"Coluna de preço alvo não encontrada para {ticker}: {target_price_col}")

    price_cols = [col for col in df.columns if col.startswith(config.PRICE_COLUMN_PREFIX)]

    df["valor_acao"] = df[target_price_col]
    df["ma7"] = df[target_price_col].rolling(window=7).mean()
    df["ma21"] = df[target_price_col].rolling(window=21).mean()

    for w in windows:
        df[f"target_greater_in_{w}d"] = (df[target_price_col].shift(-w) > df[target_price_col]).astype(int) #* config.PRICE_INCREASE_THRESHOLD
        df[f"var_acao_{w}d"] = df[target_price_col].pct_change(w)
        df[f"var_ma7_{w}d"] = df["ma7"].pct_change(w)
        df[f"var_ma21_{w}d"] = df["ma21"].pct_change(w)

        for price_col in price_cols:
            df[f"var_{price_col}_{w}d"] = df[price_col].pct_change(w)

        macro_var_cols = []
        for col in macro_cols:
            if col not in df.columns:
                continue
            var_name = f"var_{col}_{w}d"
            df[var_name] = df[col].pct_change(w)
            macro_var_cols.append(var_name)

        if macro_var_cols:
            df[f"avg_macro_var_{w}d"] = df[macro_var_cols].mean(axis=1)
            epsilon = 1e-9
            avg_macro = df[f"avg_macro_var_{w}d"].replace(0, epsilon)
            df[f"ratio_acao_vs_macro_{w}d"] = df[f"var_acao_{w}d"] / avg_macro
            df[f"ratio_acao_vs_macro_{w}d"] = (
                df[f"ratio_acao_vs_macro_{w}d"].replace([np.inf, -np.inf], np.nan).fillna(0)
            )

    max_window = max(windows)
    df_model = df.dropna(subset=[f"target_greater_in_{max_window}d", f"var_acao_{max_window}d"]).copy()
    return df_model


def get_feature_columns(df_model: pd.DataFrame) -> list[str]:
    prefixes = ("var_", "ratio_", "ma")
    return [col for col in df_model.columns if col.startswith(prefixes)]


def infer_horizon_from_target(target_col: str) -> int:
    match = re.search(r"_(\d+)d$", target_col)
    if not match:
        raise ValueError(f"Não foi possível inferir o horizonte a partir de '{target_col}'")
    return int(match.group(1))


def split_train_test(
    df_model: pd.DataFrame,
    target_col: str,
    split_percent: float = 0.8,
    horizon: Optional[int] = None,
):
    horizon = horizon or infer_horizon_from_target(target_col)
    df_model = df_model.sort_values("data").reset_index(drop=True)
    split_idx = int(len(df_model) * split_percent)
    train = df_model.iloc[:split_idx].copy()
    test = df_model.iloc[split_idx + horizon :].copy()

    feature_cols = get_feature_columns(df_model)
    test = test.dropna(subset=[target_col]).copy()

    X_train = train[feature_cols]
    y_train = train[target_col]

    print(f"Split: {len(train)} linhas para treino, {len(test)} linhas para teste")
    print(f"Train: {train['data'].min()}, {train['data'].max()}")
    print(f"Test: {test['data'].min()}, {test['data'].max()}")

    X_test = test[feature_cols]
    y_test = test[target_col]
    return X_train, y_train, X_test, y_test, feature_cols, train, test


def split_train_val_test(
    df_model: pd.DataFrame,
    target_col: str,
    train_percent: float = 0.6,
    val_percent: float = 0.2,
    horizon: Optional[int] = None,
):
    horizon = horizon or infer_horizon_from_target(target_col)
    df_model = df_model.sort_values("data").reset_index(drop=True)
    
    n_total = len(df_model)
    train_end = int(n_total * train_percent)
    val_end = int(n_total * (train_percent + val_percent))
    
    train = df_model.iloc[:train_end].copy()
    val = df_model.iloc[train_end + horizon : val_end].copy()
    test = df_model.iloc[val_end + horizon :].copy()
    
    feature_cols = get_feature_columns(df_model)
    val = val.dropna(subset=[target_col]).copy()
    test = test.dropna(subset=[target_col]).copy()

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    print(f"Triple Split: {len(train)} linhas para treino, {len(val)} linhas para validação, {len(test)} linhas para teste")
    print(f"Train: {train['data'].min()} to {train['data'].max()}")
    print(f"Val: {val['data'].min()} to {val['data'].max()}")
    print(f"Test: {test['data'].min()} to {test['data'].max()}")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, train, val, test


def get_scoring_row(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    as_of_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    df = df_model.sort_values("data").copy()
    if as_of_date:
        cutoff = pd.to_datetime(as_of_date)
        df = df[df["data"] <= cutoff].copy()

    if df.empty:
        raise ValueError("Nenhuma linha disponível para scoring na data solicitada")

    row = df.iloc[-1]
    return pd.DataFrame([row[feature_cols].fillna(0)]), row
