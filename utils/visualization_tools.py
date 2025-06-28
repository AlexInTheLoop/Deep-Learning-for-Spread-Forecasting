import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

def plot_model_metrics(df):
    metrics = {
        "R²": {"ascending": False, "title": "R²"},
        "MAE": {"ascending": True, "title": "MAE"},
        "RMSE": {"ascending": True, "title": "RMSE"},
    }

    fig = make_subplots(rows=2, cols=2, subplot_titles=[v["title"] for v in metrics.values()])

    for i, (metric, props) in enumerate(metrics.items()):
        row = i // 2 + 1
        col = i % 2 + 1
        df_sorted = df.sort_values(by=metric, ascending=props["ascending"])
        fig.add_trace(
            go.Bar(
                x=df_sorted.index,
                y=df_sorted[metric],
                text=[f"{v:.4f}" for v in df_sorted[metric]],
                textposition='auto',
                name=metric
            ),
            row=row,
            col=col
        )
        fig.update_xaxes(tickangle=45, row=row, col=col)
    title = "Comparaison des performances des méthodes de prédiction du spread moyen journalier"
    fig.update_layout(height=800, width=1000, title_text=title)
    return fig


def evaluate_and_plot(model, X, y, scaler_y=None, title="Modèle", history=None, paper_metrics=None, paper_daily=None,
                      y_true_daily=None, show_metrics=True):
    # Prédictions
    y_pred = model.predict(X, verbose=0)

    # Dénormalisation éventuelle
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
        y_pred = scaler_y.inverse_transform(y_pred).ravel()
    else:
        y_true = y.ravel()
        y_pred = y_pred.ravel()

    # Métriques
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    if show_metrics:
        print(f"{title} – Test | RMSE={rmse:.4f}  MAE={mae:.4f}")

    #  Plot 2 : scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "k--")
    plt.title(f"{title} – Dispersion")
    plt.xlabel("Spread réel")
    plt.ylabel("Spread prédit")
    plt.tight_layout()
    plt.show()

    # Plot 3 : historique des losses
    if history is not None and hasattr(history, "history"):
        hist = history.history
        if "loss" in hist:
            plt.figure(figsize=(8, 4))
            plt.plot(hist["loss"], label="Train loss")
            if "val_loss" in hist:
                plt.plot(hist["val_loss"], label="Val loss")
            plt.title(f"{title} – courbe des losses")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("  'loss' non présent dans l'objet History fourni.")

    if paper_daily is not None and y_true_daily is not None:
        # erreurs papier
        df_err = paper_daily.apply(lambda c: np.abs(c - y_true_daily))

        # erreur modèle
        err_model = np.abs(y_pred - y_true_daily)
        df_err[title] = err_model

        plt.figure(figsize=(10, 4))
        plt.boxplot([df_err[c] for c in df_err.columns],
                    labels=df_err.columns,
                    showmeans=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Erreur absolue")
        plt.title("Box-plot erreurs journalières")
        plt.tight_layout();
        plt.show()

    df_model = pd.DataFrame(
        {title: {"RMSE": rmse, "MAE": mae}}
    ).T

    if paper_metrics is not None:
        # aligne colonnes éventuelles manquantes
        df_compare = pd.concat([paper_metrics, df_model], axis=0, sort=False)
    else:
        df_compare = df_model

    # Affiche le tableau
    if show_metrics:
        display(df_compare)

    return df_compare