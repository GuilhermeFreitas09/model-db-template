from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

from simple_quant import config


class MiningDataBuilder:
    def __init__(self, start_date: str = config.DEFAULT_START_DATE):
        self.start_dt = pd.to_datetime(start_date)
        self.start_date_bcb = self.start_dt.strftime("%d/%m/%Y")
        self.end_date_bcb = datetime.now().strftime("%d/%m/%Y")
        self.tickers_mining = config.TICKERS_MINING
        self.yfinance_features = config.YFINANCE_FEATURES
        self.bcb_features = config.BCB_FEATURES
        self.tickers_macro = list(self.yfinance_features.keys())
        self.all_tickers = self.tickers_mining + self.tickers_macro

    def get_market_data(self) -> pd.DataFrame:
        df = yf.download(self.all_tickers, start=self.start_dt, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        df = df.reset_index()

        rename_map = {"Date": "data", **self.yfinance_features}
        rename_map.update({ticker: config.price_column_name(ticker) for ticker in self.tickers_mining})
        df.rename(columns=rename_map, inplace=True)
        return df

    def get_bcb_series(self, code: int, name: str) -> pd.DataFrame:
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
            f"?formato=json&dataInicial={self.start_date_bcb}&dataFinal={self.end_date_bcb}"
        )
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(columns=["data", name])

        df["data"] = pd.to_datetime(df["data"], dayfirst=True)
        df[name] = pd.to_numeric(df["valor"])
        return df[["data", name]]

    def build(self) -> pd.DataFrame:
        market_df = self.get_market_data()
        if market_df.empty:
            raise ValueError("Nenhum dado de mercado foi retornado")

        df_final = market_df.copy()
        for feature_name, feature_code in self.bcb_features.items():
            feature_df = self.get_bcb_series(feature_code, feature_name)
            df_final = pd.merge(df_final, feature_df, on="data", how="left")

        df_final = df_final.sort_values("data").ffill()

        df_final['ts'] = datetime.now()
        
        return df_final
