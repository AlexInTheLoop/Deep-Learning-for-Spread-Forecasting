import os, numpy as np, pandas as pd
from data.DataManager import DataManager
from utils.res_comp import compute_parametric_estimators
from utils.metrics import compute_estimators_metrics 


class PaperEstimatorsRunner:
    """ Classe pour charger les estimateurs du papier et évaluer les performances sur les données de Binance."""

    def __init__(self, symbols,periods,*,use_opposed=False,light_download=True,auto_labels=True):
             
        self.symbols = [s.upper() for s in symbols]
        self.periods = periods
                    
        if len(self.symbols) != len(self.periods):
            raise ValueError("len(symbols) doit == len(periods)")
        self.tasks = [(s, y, m) for (s, (y, m)) in zip(self.symbols,self.periods)]
                                                            
        self._dms = {}
        for sym, y, m in self.tasks:
            self._dms.setdefault((y, m), []).append(sym)

        self._dms = {
            (y, m): DataManager(
                symbols = syms,         
                dates   = [(y, m)],      
                light   = light_download
            )
            for (y, m), syms in self._dms.items()
        }

        self.use_opposed  = use_opposed
        self.auto_labels  = auto_labels
        self._df_est    = None
        self._df_labels = None

    def _load_klines_one_month(self, dm, year, month, symbol):
        """Charge les données de kline pour un symbole et une période donnés."""


        fname = f"{symbol}-1m-{year}-{month:02d}.parquet"
        path  = os.path.join(dm.raw_data_dir, fname)
        df = pd.read_parquet(path,
                             columns=["open_time", "open", "high", "low", "close"])
        df["day"]  = pd.to_datetime(df["open_time"], unit="ms").dt.day
        return df

    def get_estimates(self, force=False):
        """Calcule/charge les estimateurs pour toutes les périodes & actifs."""

        if self._df_est is not None and not force:
            return self._df_est    

        # Si les labels sont activés, on charge les labels
        frames = []
        for (y, m), dm in self._dms.items():
            dm.download_and_prepare_data()        

            # Charge les données de kline pour chaque symbole et période
            for sym in self.symbols:
                df = self._load_klines_one_month(dm, y, m, sym)
                for d, grp in df.groupby("day", sort=True):
                    est = compute_parametric_estimators( grp[["open", "high", "low", "close"]],use_opposed=self.use_opposed)

                    frames.append({
                        "symbol": sym, "year": y, "month": m, "day": int(d),
                        "S1": est[0], "S2": est[1], "S3": est[2], "S4": est[3],
                        "Roll Spread": est[4], "CS Spread": est[5],
                        "AGK1": est[6], "AGK2": est[7], "AR Spread": est[8]
                    })

        self._df_est = (pd.DataFrame(frames) .set_index(["symbol", "year", "month", "day"]).sort_index())
            
        return self._df_est

    def _load_labels_all(self):
        """
        Construit (si nécessaire) le fichier des labels en ne lisant
        que les trois colonnes, puis charge tous les labels.
        """
        if self._df_labels is not None:
            return self._df_labels

        frames = []

        for sym, y, m in self.tasks:
            dm = self._dms[(y, m)]
            raw_path = os.path.join(
                dm.raw_data_dir,
                f"{sym}-bookTicker-{y}-{m:02d}.parquet"
            )
            lab_path = os.path.join(
                dm.labels_data_dir,
                f"{sym}-bookTicker-{y}-{m:02d}_labels.parquet"
            )
            os.makedirs(dm.labels_data_dir, exist_ok=True)

            if not os.path.exists(lab_path):
                if not os.path.exists(raw_path):
                    raise FileNotFoundError(f"Parquet bookTicker manquant : {raw_path}")

                df_bt = pd.read_parquet(raw_path,columns=["transaction_time","best_bid_price","best_ask_price"])
                df_bt["datetime"] = pd.to_datetime( df_bt["transaction_time"].astype(np.int64), unit="ms")
                df_bt["day"] = df_bt["datetime"].dt.day
                df_bt["spread"] = (df_bt["best_ask_price"] - df_bt["best_bid_price"])
                df_lab = (df_bt.groupby("day")["spread"].mean().reset_index().rename(columns={"spread": "spread_real"}))     
                df_lab.to_parquet(lab_path, index=False)

            df = pd.read_parquet(lab_path)
            df["symbol"] = sym
            df["year"]   = y
            df["month"]  = m
            frames.append(df)

        self._df_labels = (pd.concat(frames, ignore_index=True).set_index(["symbol", "year", "month", "day"]).sort_index())
                
        return self._df_labels

    def evaluate(self, y_metric="all",sort_mode="asset_first"):  
        """ Calcule les métriques (R2, RMSE, MAE) par estimate"""

        df_est   = self.get_estimates()
        df_lab   = self._load_labels_all()
        df_join  = df_est.join(df_lab, how="inner")

        y_true   = df_join["spread_real"].values
        df_pred  = df_join.drop(columns="spread_real")

        df_scores = (compute_estimators_metrics(df_pred,y_true,sort_mode=sort_mode).T)

        if y_metric != "all":
            if isinstance(y_metric, str):
                df_scores = df_scores[[y_metric]]
            else:
                df_scores = df_scores[y_metric]
        return df_scores