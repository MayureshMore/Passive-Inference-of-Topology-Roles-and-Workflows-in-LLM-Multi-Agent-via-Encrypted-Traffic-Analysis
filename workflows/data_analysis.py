"""
Data analysis workflow — structured data (CSV-like) analysed and summarised.
Characteristic traffic: larger initial payload (the data), smaller streaming
responses as the executor works through rows, then a validator summary.
"""

from __future__ import annotations

import random

from .base import BaseWorkflow, WorkflowClass

# Larger, more varied datasets — 15–25 rows each to drive bigger request payloads
_DATASETS = [
    ("sales_q3.csv",
     "date,product,region,units,revenue\n"
     "2024-07-01,WidgetA,North,120,4800\n2024-07-01,WidgetB,South,85,5100\n"
     "2024-07-02,WidgetA,East,95,3800\n2024-07-02,WidgetC,West,210,12600\n"
     "2024-07-03,WidgetB,North,130,7800\n2024-07-03,WidgetA,South,145,5800\n"
     "2024-07-04,WidgetC,East,88,5280\n2024-07-04,WidgetB,West,92,5520\n"
     "2024-07-05,WidgetA,North,178,7120\n2024-07-05,WidgetC,South,65,3900\n"
     "2024-07-06,WidgetB,East,112,6720\n2024-07-06,WidgetA,West,99,3960\n"
     "2024-07-07,WidgetC,North,234,14040\n2024-07-07,WidgetB,South,76,4560\n"
     "2024-07-08,WidgetA,East,156,6240\n2024-07-08,WidgetC,West,189,11340\n"
     "2024-07-09,WidgetB,North,143,8580\n2024-07-09,WidgetA,South,201,8040\n"
     "2024-07-10,WidgetC,East,77,4620\n2024-07-10,WidgetB,West,118,7080\n"),

    ("server_metrics.csv",
     "timestamp,host,cpu_pct,mem_pct,disk_io_mbps,net_mbps,latency_ms,errors\n"
     "2024-08-15T10:00,web-01,42.1,68.3,12.4,45.2,12,0\n"
     "2024-08-15T10:00,web-02,78.9,71.0,8.1,120.5,45,3\n"
     "2024-08-15T10:05,web-01,91.2,72.1,55.3,89.1,120,12\n"
     "2024-08-15T10:05,db-01,35.0,88.7,210.4,18.2,8,0\n"
     "2024-08-15T10:10,web-02,88.4,74.2,9.2,98.3,98,7\n"
     "2024-08-15T10:10,web-01,45.3,69.8,14.1,52.3,15,0\n"
     "2024-08-15T10:15,db-01,42.1,91.3,198.7,22.1,11,1\n"
     "2024-08-15T10:15,web-02,95.1,76.4,7.8,145.6,210,28\n"
     "2024-08-15T10:20,web-01,38.7,70.1,11.2,48.9,13,0\n"
     "2024-08-15T10:20,cache-01,12.4,45.2,2.1,890.4,1,0\n"
     "2024-08-15T10:25,web-02,82.3,77.8,9.9,134.2,67,5\n"
     "2024-08-15T10:25,db-01,38.9,92.1,245.3,19.8,14,2\n"
     "2024-08-15T10:30,web-01,99.8,73.4,58.1,78.2,450,45\n"
     "2024-08-15T10:30,cache-01,14.1,46.8,2.3,920.1,1,0\n"
     "2024-08-15T10:35,web-02,71.2,75.3,8.4,112.7,52,2\n"),

    ("user_churn.csv",
     "user_id,signup_date,last_active,plan,sessions_30d,pages_30d,support_tickets,churned\n"
     "U001,2023-01-10,2024-06-20,pro,45,892,1,0\n"
     "U002,2023-03-15,2023-11-05,free,2,18,0,1\n"
     "U003,2022-11-01,2024-07-28,enterprise,128,4521,3,0\n"
     "U004,2024-02-20,2024-03-01,free,1,4,2,1\n"
     "U005,2023-08-12,2024-07-10,pro,32,678,0,0\n"
     "U006,2023-05-20,2024-01-15,free,5,42,1,1\n"
     "U007,2022-06-01,2024-07-29,enterprise,210,8934,2,0\n"
     "U008,2024-01-05,2024-02-28,pro,8,145,4,1\n"
     "U009,2023-11-30,2024-07-25,free,18,312,0,0\n"
     "U010,2023-07-14,2024-07-30,pro,67,1823,1,0\n"
     "U011,2024-03-01,2024-04-10,free,3,28,3,1\n"
     "U012,2022-09-15,2024-07-28,enterprise,156,5672,0,0\n"
     "U013,2023-12-01,2024-06-30,pro,22,445,2,0\n"
     "U014,2024-04-15,2024-05-20,free,4,35,1,1\n"
     "U015,2023-02-28,2024-07-15,pro,41,967,0,0\n"),

    ("ab_test_results.csv",
     "experiment_id,variant,user_segment,impressions,clicks,conversions,revenue_usd\n"
     "EXP-001,control,new_users,12450,892,134,2680.00\n"
     "EXP-001,treatment_a,new_users,12380,1124,198,3960.00\n"
     "EXP-001,treatment_b,new_users,12510,1045,167,3340.00\n"
     "EXP-001,control,returning,8920,1456,312,9360.00\n"
     "EXP-001,treatment_a,returning,8870,1678,389,11670.00\n"
     "EXP-002,control,mobile,25100,1820,245,4900.00\n"
     "EXP-002,treatment,mobile,24980,2340,378,7560.00\n"
     "EXP-002,control,desktop,18450,2890,567,17010.00\n"
     "EXP-002,treatment,desktop,18520,3120,634,19020.00\n"
     "EXP-003,control,all,45200,4580,890,26700.00\n"
     "EXP-003,treatment_a,all,44980,5120,1034,31020.00\n"
     "EXP-003,treatment_b,all,45100,4890,967,29010.00\n"
     "EXP-003,treatment_c,all,45050,5340,1123,33690.00\n"),

    ("inventory_forecast.csv",
     "sku,category,warehouse,stock_on_hand,reorder_point,lead_days,"
     "avg_daily_demand,demand_std,stockout_risk_pct,days_of_supply\n"
     "SKU-A001,Electronics,WH-East,450,200,14,28.5,8.2,2.1,15.8\n"
     "SKU-A002,Electronics,WH-East,89,150,14,32.1,12.4,34.5,2.8\n"
     "SKU-B001,Apparel,WH-West,2340,500,7,145.2,45.1,0.3,16.1\n"
     "SKU-B002,Apparel,WH-West,312,400,7,89.4,28.7,18.2,3.5\n"
     "SKU-C001,HomeGoods,WH-North,1890,800,21,67.3,15.2,1.2,28.1\n"
     "SKU-C002,HomeGoods,WH-North,445,600,21,78.9,22.4,12.8,5.6\n"
     "SKU-D001,Food,WH-South,678,1000,3,234.5,67.8,8.9,2.9\n"
     "SKU-D002,Food,WH-South,1234,1000,3,198.2,45.1,0.8,6.2\n"
     "SKU-E001,Electronics,WH-West,234,300,21,15.6,4.2,22.4,15.0\n"
     "SKU-E002,Apparel,WH-East,1567,400,7,112.3,34.5,0.1,13.9\n"),

    ("marketing_funnel.csv",
     "channel,campaign,week,impressions,clicks,leads,trials,paid_conversions,cac_usd,ltv_usd\n"
     "Google,Brand,W01,145200,8920,1234,345,89,112.40,890.00\n"
     "Google,NonBrand,W01,892400,12450,890,156,34,298.50,670.00\n"
     "LinkedIn,Sponsored,W01,45200,1230,456,123,45,189.20,1240.00\n"
     "Email,Newsletter,W01,89000,4560,678,234,112,22.50,780.00\n"
     "Google,Brand,W02,152400,9340,1456,389,102,108.90,910.00\n"
     "Google,NonBrand,W02,934000,13120,945,178,41,285.10,680.00\n"
     "LinkedIn,Sponsored,W02,48900,1456,512,145,52,178.50,1290.00\n"
     "Email,Newsletter,W02,91200,4890,712,256,128,20.10,800.00\n"
     "Organic,SEO,W01,234000,18900,2340,567,234,0.00,780.00\n"
     "Organic,SEO,W02,248000,20100,2560,612,267,0.00,810.00\n"
     "Referral,Partner,W01,12400,890,345,189,98,45.20,1450.00\n"
     "Referral,Partner,W02,13800,978,389,212,115,42.80,1520.00\n"),
]

_ANALYSIS_ASKS = [
    "Perform a comprehensive analysis: identify trends, anomalies, outliers, and key statistics. Provide actionable recommendations.",
    "Compute detailed summary statistics (mean, median, std, min, max, percentiles) for all numeric columns. Flag data quality issues.",
    "What are the top 3 business insights from this dataset? Support each with specific numbers from the data.",
    "Perform comparative analysis across all categories/groups. Rank them and explain what drives the differences.",
    "Identify the best and worst performers. Quantify the performance gap and hypothesize root causes.",
    "Build a simple forecast or projection based on the trends in this data. State your assumptions clearly.",
    "Detect correlations between variables. Which factors most strongly predict the target metric?",
    "Flag any data quality issues, missing values, or anomalies that would affect downstream analysis.",
    "Segment the data into meaningful groups and characterise each segment with supporting statistics.",
    "Compute month-over-month or period-over-period changes. Which metrics show statistically significant shifts?",
]


class DataAnalysisWorkflow(BaseWorkflow):
    workflow_class = WorkflowClass.DATA_ANALYSIS

    def generate_prompt(self) -> str:
        name, data = random.choice(_DATASETS)
        ask = random.choice(_ANALYSIS_ASKS)
        return f"{ask}\n\nDataset: {name}\n\n```csv\n{data}```"
