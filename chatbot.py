# chatbot.py
def chatbot_response(df, query):
    q = query.lower()

    if "missed" in q:
        return f"Total missed vaccinations: {(df['status']=='missed').sum()}"

    if "pending" in q:
        return f"Total pending vaccinations: {(df['status']=='pending').sum()}"

    if "completed" in q:
        return f"Total completed vaccinations: {(df['status']=='completed').sum()}"

    if "top district" in q:
        d = df.groupby("district").size().sort_values(ascending=False).head(1)
        return f"Top district: {d.index[0]} ({d.iloc[0]} records)"

    if "top state" in q:
        s = df.groupby("state").size().sort_values(ascending=False).head(1)
        return f"Top state: {s.index[0]} ({s.iloc[0]} records)"

    return "Ask about missed, pending, completed, top district, or top state."
