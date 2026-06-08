"""
Data analysis workflow — structured data (CSV-like) analysed and summarised.
Characteristic traffic: larger initial payload (the data), smaller streaming
responses as the executor works through rows, then a validator summary.
"""

from __future__ import annotations

import random

from .base import BaseWorkflow, WorkflowClass

_DATASETS = [
    ("sales_q3.csv", "date,product,region,units,revenue\n"
     "2024-07-01,WidgetA,North,120,4800\n"
     "2024-07-01,WidgetB,South,85,5100\n"
     "2024-07-02,WidgetA,East,95,3800\n"
     "2024-07-02,WidgetC,West,210,12600\n"
     "2024-07-03,WidgetB,North,130,7800\n"),

    ("server_metrics.csv", "timestamp,host,cpu_pct,mem_pct,latency_ms\n"
     "2024-08-15T10:00,web-01,42.1,68.3,12\n"
     "2024-08-15T10:00,web-02,78.9,71.0,45\n"
     "2024-08-15T10:05,web-01,91.2,72.1,120\n"
     "2024-08-15T10:05,db-01,35.0,88.7,8\n"
     "2024-08-15T10:10,web-02,88.4,74.2,98\n"),

    ("user_churn.csv", "user_id,signup_date,last_active,plan,churned\n"
     "U001,2023-01-10,2024-06-20,pro,0\n"
     "U002,2023-03-15,2023-11-05,free,1\n"
     "U003,2022-11-01,2024-07-28,enterprise,0\n"
     "U004,2024-02-20,2024-03-01,free,1\n"
     "U005,2023-08-12,2024-07-10,pro,0\n"),
]

_ANALYSIS_ASKS = [
    "Identify trends, anomalies, and key statistics in this dataset.",
    "Compute summary statistics and flag any outliers or data quality issues.",
    "What business insights can you draw from this data? Highlight top 3.",
    "Perform a comparative analysis across the categories in this dataset.",
    "Identify the top performers and bottom performers, explain the gap.",
]


class DataAnalysisWorkflow(BaseWorkflow):
    workflow_class = WorkflowClass.DATA_ANALYSIS

    def generate_prompt(self) -> str:
        name, data = random.choice(_DATASETS)
        ask = random.choice(_ANALYSIS_ASKS)
        return f"{ask}\n\nDataset: {name}\n\n```csv\n{data}```"
