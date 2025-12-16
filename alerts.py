# alerts.py
def high_missed_districts(df, threshold=0.3):
    summary = (
        df.groupby("district")["status"]
        .apply(lambda x: (x == "missed").mean())
        .reset_index(name="missed_rate")
    )
    return summary[summary["missed_rate"] > threshold]
