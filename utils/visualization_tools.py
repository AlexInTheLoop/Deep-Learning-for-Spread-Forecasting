import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_model_metrics(df):
    metrics = {
        "R²": {"ascending": False, "title": "R²"},
        "MAE": {"ascending": True, "title": "MAE"},
        "RMSE": {"ascending": True, "title": "RMSE"},
        "Score": {"ascending": False, "title": "Score"}
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