
import plotly.graph_objects as go
import pandas as pd

def plot_history_forecast(history: pd.Series, forecast: pd.Series, lower=None, upper=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, name="History", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast", mode="lines"))
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            name="80% interval",
            line=dict(width=0),
            opacity=0.2
        ))
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
