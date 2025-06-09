import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def create_dynamic_visualizations(df: pd.DataFrame, column_analysis: dict) -> list:
    """Create relevant visualizations based on detected column types"""
    figures = []
    temporal_cols = [col for col, info in column_analysis.items() if info['likely_purpose'] == 'temporal']
    financial_cols = [col for col, info in column_analysis.items() if info['likely_purpose'] == 'financial']
    metric_cols = [col for col, info in column_analysis.items() if info['likely_purpose'] == 'performance_metric']

    if temporal_cols and (financial_cols or metric_cols):
        for time_col in temporal_cols[:1]:
            try:
                df_sorted = df.sort_values(time_col)
                if financial_cols:
                    fig = go.Figure()
                    for fin_col in financial_cols[:3]:
                        if pd.api.types.is_numeric_dtype(df_sorted[fin_col]):
                            fig.add_trace(go.Scatter(
                                x=df_sorted[time_col],
                                y=df_sorted[fin_col],
                                mode='lines+markers',
                                name=fin_col,
                                line=dict(width=2)
                            ))
                    if fig.data:
                        fig.update_layout(
                            title="Financial Metrics Over Time",
                            xaxis_title=time_col,
                            yaxis_title="Value",
                            hovermode='x unified'
                        )
                        figures.append(fig)
                if metric_cols:
                    fig = go.Figure()
                    for metric_col in metric_cols[:3]:
                        if pd.api.types.is_numeric_dtype(df_sorted[metric_col]):
                            fig.add_trace(go.Scatter(
                                x=df_sorted[time_col],
                                y=df_sorted[metric_col],
                                mode='lines+markers',
                                name=metric_col,
                                line=dict(width=2)
                            ))
                    if fig.data:
                        fig.update_layout(
                            title="Performance Metrics Over Time",
                            xaxis_title=time_col,
                            yaxis_title="Value",
                            hovermode='x unified'
                        )
                        figures.append(fig)
            except Exception as e:
                # In Streamlit, warnings happen in st.warning calls; here, propagate exception
                raise RuntimeError(f"Could not create time series for {time_col}: {e}")

    numeric_cols = [col for col, info in column_analysis.items()
                    if info['likely_purpose'] in ['financial', 'performance_metric', 'volume_metric', 'numeric']
                    and pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) > 2:
        try:
            numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Metric Correlations",
                    color_continuous_scale='RdBu_r'
                )
                figures.append(fig)
        except Exception as e:
            raise RuntimeError(f"Could not create correlation matrix: {e}")

    return figures