import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from groq import Groq
from newsapi import NewsApiClient
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

df = yf.download("GC=F", start="2020-01-01")
df = df["Close"].reset_index()
df.columns = ["ds", "y"]
df["unique_id"] = "gold"
df = df.dropna()

PDF_PATHS = [os.getenv(f"PDF_{i}") for i in range(1, 8) if os.getenv(f"PDF_{i}")]

print("loading pdfs...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_chunks = []
for path in PDF_PATHS:
    loader = PyPDFLoader(path)
    pages = loader.load()
    chunks = splitter.split_documents(pages)
    all_chunks.extend(chunks)

print(f"{len(all_chunks)} chunks loaded")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("gold_rag")
texts = [c.page_content for c in all_chunks]
ids = [str(i) for i in range(len(texts))]
embeds = embeddings.embed_documents(texts)
collection.add(documents=texts, embeddings=embeds, ids=ids)
print("rag ready")

def get_rag_context(query, n=3):
    q = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[q], n_results=n)
    return "\n".join(results["documents"][0])

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def get_news(date):
    date_str = str(date.date()) if hasattr(date, "date") else str(date)
    articles = newsapi.get_everything(
        q="gold price market federal reserve",
        from_param=date_str,
        to=date_str,
        language="en",
        sort_by="relevancy",
        page_size=3
    )
    headlines = [a["title"] for a in articles["articles"]]
    return "\n".join(headlines) if headlines else "no news found"

def get_news_range(days_back=7):
    today = datetime.today()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    articles = newsapi.get_everything(
        q="gold price market federal reserve",
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="relevancy",
        page_size=5
    )
    headlines = [a["title"] for a in articles["articles"]]
    return "\n".join(headlines) if headlines else "no news found"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

while True:
    try:
        print("\nGC=F Gold Forecasting CLI")
        print("1) Direct Forecast")
        print("2) 30-Day Anomaly Analysis")
        print("3) Holdout Backtest")
        print("q) Quit")
        choice = input("select: ").strip()

        if choice == "q":
            break

        if choice == "1":
            train_points = int(input("train points (e.g. 500): "))
            horizon = int(input("horizon days (e.g. 30): "))
            input_size = int(input("input size (e.g. 60): "))
            n_heads = int(input("n_heads (e.g. 4): "))
            patch_len = int(input("patch_len (e.g. 16): "))

            train_df = df.tail(train_points).reset_index(drop=True)
            model = NeuralForecast(
                models=[PatchTST(h=horizon, input_size=input_size, max_steps=500,
                                 n_heads=n_heads, patch_len=patch_len)],
                freq="B"
            )
            model.fit(train_df)
            preds = model.predict().reset_index(drop=True)
            preds["ds"] = pd.to_datetime(preds["ds"])

            plt.figure(figsize=(14, 5))
            plt.plot(df["ds"].tail(100), df["y"].tail(100), "b-o", label="actual (last 100)")
            plt.plot(preds["ds"], preds["PatchTST"], "g-s", label="forecast")
            plt.axvline(df["ds"].iloc[-1], color="orange", linestyle="--", label="today")
            plt.title(f"GC=F forecast ({horizon}d)")
            plt.xlabel("date")
            plt.ylabel("price")
            plt.legend()
            plt.tight_layout()
            plt.show()

            news = get_news_range(days_back=7)
            start_price = df["y"].iloc[-1]
            end_price = preds["PatchTST"].iloc[-1]
            change_pct = (end_price - start_price) / start_price * 100
            rag = get_rag_context(
                f"gold price forecast risk factors federal reserve monetary policy "
                f"geopolitical uncertainty inflation dollar index "
                f"current price ${start_price:.0f} direction {'bullish' if end_price > start_price else 'bearish'}"
            )
            prompt = f"""
Gold forecast next {horizon} business days.
Current: ${start_price:.0f} | Forecast: ${end_price:.0f} | Direction: {"UP" if end_price > start_price else "DOWN"} ({change_pct:+.1f}%)

News (last 7d):
{news}

Theory:
{rag}

What are the main risks that could invalidate this forecast?
"""
            print(ask_llm(prompt))

        elif choice == "2":
            train_points = int(input("train points (e.g. 500): "))
            input_size = int(input("input size (e.g. 60): "))
            n_heads = int(input("n_heads (e.g. 4): "))
            patch_len = int(input("patch_len (e.g. 16): "))

            today = datetime.today()
            start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")

            actual = yf.download("GC=F", start=start_date)
            actual = actual["Close"].reset_index()
            actual.columns = ["ds", "actual"]
            actual["ds"] = pd.to_datetime(actual["ds"]) 

            split_idx = len(df) - len(actual)
            train_df = df.iloc[max(0, split_idx - train_points):split_idx].reset_index(drop=True)

            model = NeuralForecast(
                models=[PatchTST(h=len(actual), input_size=input_size, max_steps=500,
                                 n_heads=n_heads, patch_len=patch_len)],
                freq="B"
            )
            model.fit(train_df)
            preds = model.predict().reset_index(drop=True)
            preds["ds"] = pd.to_datetime(preds["ds"])

            result = preds.merge(actual, on="ds")
            result["rolling_mean"] = result["actual"].rolling(window=5, min_periods=1).mean().shift(1)
            result["deviation_pct"] = abs(result["actual"] - result["rolling_mean"]) / result["rolling_mean"] * 100
            result["error"] = abs(result["actual"] - result["PatchTST"])
            result["error_pct"] = result["error"] / result["actual"] * 100
            result["is_anomaly"] = result["error_pct"] > 1.5
            result["anomaly_group"] = (result["is_anomaly"] != result["is_anomaly"].shift()).cumsum()
            anomaly_events = result[result["is_anomaly"]].groupby("anomaly_group").first().reset_index(drop=True)

            rmse = np.sqrt(((result["actual"] - result["PatchTST"]) ** 2).mean())
            print(f"rmse: {rmse:.3f} | anomalies: {len(anomaly_events)}")

            plt.figure(figsize=(14, 5))
            plt.plot(result["ds"], result["actual"], "b-o", label="actual")
            plt.plot(result["ds"], result["PatchTST"], "g-s", label="forecast")
            for _, row in anomaly_events.iterrows():
                plt.axvline(row["ds"], color="red", linestyle="--", alpha=0.6)
            plt.title(f"GC=F anomaly analysis | rmse={rmse:.3f} | threshold=3%")
            plt.xlabel("date")
            plt.ylabel("price")
            plt.legend()
            plt.tight_layout()
            plt.show()

            for _, row in anomaly_events.iterrows():
                news = get_news(row["ds"])
                rag = get_rag_context(
                    f"gold price movement federal reserve interest rate geopolitical risk "
                    f"inflation dollar correlation technical breakout "
                    f"price ${row['actual']:.0f} deviation {row['deviation_pct']:.1f}%"
                )
                prompt = f"""
Gold anomaly on {row['ds'].date()}.
Actual: ${row['actual']:.0f} | 5d avg: ${row['rolling_mean']:.0f} | error: {row['error_pct']:.1f}% (${row['error']:.0f})

News:
{news}

Theory:
{rag}

Why did this anomaly occur?
"""
                print(f"\n{row['ds'].date()} — error: {row['error_pct']:.1f}% (${row['error']:.0f})")
                print(f"news: {news}")
                print(f"explanation: {ask_llm(prompt)}")

        elif choice == "3":
            offset_back = int(input("offset from end (e.g. 500): "))
            train_points = int(input("train points (e.g. 3000): "))
            horizon = int(input("horizon (e.g. 90): "))
            input_size = int(input("input size (e.g. 60): "))
            n_heads = int(input("n_heads (e.g. 4): "))
            patch_len = int(input("patch_len (e.g. 16): "))

            split_idx = len(df) - offset_back
            train_df = df.iloc[split_idx - train_points:split_idx].reset_index(drop=True)
            actual_df = df.iloc[split_idx:split_idx + horizon].reset_index(drop=True)

            model = NeuralForecast(
                models=[PatchTST(h=horizon, input_size=input_size, max_steps=500,
                                 n_heads=n_heads, patch_len=patch_len)],
                freq="B"
            )
            model.fit(train_df)
            preds = model.predict().reset_index(drop=True)
            preds["ds"] = pd.to_datetime(preds["ds"])
            actual_df["ds"] = pd.to_datetime(actual_df["ds"])
            result = preds.merge(actual_df[["ds", "y"]], on="ds", how="inner")

            rmse = np.sqrt(((result["y"] - result["PatchTST"]) ** 2).mean())
            print(f"rmse: {rmse:.3f}")

            plt.figure(figsize=(14, 5))
            plt.plot(result["ds"], result["y"], "b-o", label="actual")
            plt.plot(result["ds"], result["PatchTST"], "g-s", label="forecast")
            plt.title(f"GC=F backtest | rmse={rmse:.3f}")
            plt.xlabel("date")
            plt.ylabel("price")
            plt.legend()
            plt.tight_layout()
            plt.show()

    except ValueError as e:
        print(f"error: {e}")