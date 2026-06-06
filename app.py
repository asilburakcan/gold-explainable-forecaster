
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

# ─── 1. VERİ (bugüne kadar) ───
df = yf.download("GC=F", start="2020-01-01")
df = df["Close"].reset_index()
df.columns = ["ds", "y"]
df["unique_id"] = "gold"
df = df.dropna()

# ─── 2. RAG (bir kez yüklenir) ───
PDF_PATHS = [v for k, v in sorted(os.environ.items()) if k.startswith("PDF_")]


print("PDF'ler yükleniyor...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_chunks = []
for path in PDF_PATHS:
    loader = PyPDFLoader(path)
    pages = loader.load()
    chunks = splitter.split_documents(pages)
    all_chunks.extend(chunks)

print(f"{len(all_chunks)} chunk oluşturuldu, ChromaDB'ye yükleniyor...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("gold_rag")
texts  = [c.page_content for c in all_chunks]
ids    = [str(i) for i in range(len(texts))]
embeds = embeddings.embed_documents(texts)
collection.add(documents=texts, embeddings=embeds, ids=ids)
print("RAG hazır.")

def get_rag_context(query, n=3):
    query_embed = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_embed], n_results=n)
    return "\n".join(results["documents"][0])

# ─── 3. HABER ───
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
    return "\n".join(headlines) if headlines else "No news found."

def get_news_range(days_back=7):
    today = datetime.today()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date   = today.strftime("%Y-%m-%d")
    articles = newsapi.get_everything(
        q="gold price market federal reserve",
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="relevancy",
        page_size=5
    )
    headlines = [a["title"] for a in articles["articles"]]
    return "\n".join(headlines) if headlines else "No news found."

# ─── 4. LLM ───
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
        # ─── 5. CLI ───
        print("\nGC=F Gold Forecasting CLI")
        print("1) Direct Forecast (Future)")
        print("2) 30-Day Anomaly Analysis")
        print("3) Holdout Backtest")
        print("q) Quit")
        choice = input("Select (1/2/3/q): ").strip()

        if choice == "q":
            break

        # ─────────────────────────────────────
        if choice == "1":
            print("\n─── DIRECT FORECAST (FUTURE) ───")
            train_points = int(input("Training points before how many days? (e.g. 500): "))
            horizon    = int(input("How many days into the future? (e.g. 30): "))
            input_size = int(input("Input (lag window) size? (e.g. 60): "))
            max_steps  = 50
            n_heads    = int(input("n_heads? (e.g. 4): "))
            patch_len  = int(input("patch_len? (e.g. 16): "))

            train_df = df.tail(train_points).reset_index(drop=True)
            model = NeuralForecast(
                models=[PatchTST(h=horizon, input_size=input_size, max_steps=max_steps, n_heads=n_heads, patch_len=patch_len)],
                freq="B"
            )
            model.fit(train_df)
            preds = model.predict().reset_index(drop=True)
            preds["ds"] = pd.to_datetime(preds["ds"])

            plt.figure(figsize=(14, 5))
            plt.plot(df["ds"].tail(100), df["y"].tail(100), "b-o", label="Actual (last 100)")
            plt.plot(preds["ds"], preds["PatchTST"], "g-s", label="Forecast")
            plt.axvline(df["ds"].iloc[-1], color="orange", linestyle="--", label="Today")
            plt.title(f"GC=F - Future Forecast ({horizon} days)")
            plt.xlabel("Date"); plt.ylabel("Price"); plt.legend()
            plt.tight_layout(); plt.show()

            print("\nRisk analizi üretiliyor...")
            news = get_news_range(days_back=7)
            start_price = df["y"].iloc[-1]
            end_price   = preds["PatchTST"].iloc[-1]
            change_pct = (end_price - start_price) / start_price * 100
            rag  = get_rag_context( f"gold price forecast risk factors: "
                                    f"federal reserve monetary policy outlook, "
                                    f"geopolitical uncertainty safe haven flows, "
                                    f"inflation expectations gold hedge, "
                                    f"dollar index inverse correlation gold, "
                                    f"current price: ${start_price:.0f}, "
                                    f"forecast direction: {'bullish' if end_price > start_price else 'bearish'}")
            prompt = f"""
            Gold price forecast for the next {horizon} business days:
            Current price: ${start_price:.0f}
            Forecasted price: ${end_price:.0f}
            Direction: {"UP" if end_price > start_price else "DOWN"} ({change_pct:+.1f}%)
            

            Recent news (last 7 days):
            {news}

            Relevant theory from technical analysis books:
            {rag}

            Based on the current market context and theory, what are the main risks
            that could invalidate this forecast? Be specific and concise.
            """
            print("\n─── RISK ANALYSIS ───")
            print(ask_llm(prompt))

        # ─────────────────────────────────────
        elif choice == "2":
            print("\n─── 30-DAY ANOMALY ANALYSIS ───")
            train_points = int(input("Training points before last 30 days? (e.g. 500): "))
            input_size   = int(input("Input (lag window) size? (e.g. 60): "))
            max_steps    = 50
            n_heads      = int(input("n_heads? (e.g. 4): "))
            patch_len    = int(input("patch_len? (e.g. 16): "))

            today      = datetime.today()
            start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")

            actual = yf.download("GC=F", start=start_date)
            actual = actual["Close"].reset_index()
            actual.columns = ["ds", "actual"]
            actual["ds"] = pd.to_datetime(actual["ds"])

            split_idx = len(df) - len(actual)
            train_df  = df.iloc[max(0, split_idx - train_points):split_idx].reset_index(drop=True)

            model = NeuralForecast(
                models=[PatchTST(h=len(actual), input_size=input_size, max_steps=max_steps, n_heads=n_heads, patch_len=patch_len)],
                freq="B"
            )
            model.fit(train_df)
            preds = model.predict().reset_index(drop=True)
            preds["ds"] = pd.to_datetime(preds["ds"])

            result = preds.merge(actual, on="ds")

            # ─── YENİ ANOMALİ TESPİT: son 5 günün ortalamasından %3 sapma ───
            result["rolling_mean"] = result["actual"].rolling(window=5, min_periods=1).mean().shift(1)
            result["deviation_pct"] = abs(result["actual"] - result["rolling_mean"]) / result["rolling_mean"] * 100
            result["error"] = abs(result["actual"] - result["PatchTST"])
            result["error_pct"] = result["error"] / result["actual"] * 100
            result["is_anomaly"] = result["error_pct"] > 3

            # Ardışık günleri grupla — tek olay olarak say
            result["anomaly_group"] = (result["is_anomaly"] != result["is_anomaly"].shift()).cumsum()
            anomaly_events = result[result["is_anomaly"]].groupby("anomaly_group").first().reset_index(drop=True)

            rmse = np.sqrt(((result["actual"] - result["PatchTST"]) ** 2).mean())
            print(f"RMSE: {rmse:.3f} | Anomaly Events: {len(anomaly_events)}")

            plt.figure(figsize=(14, 5))
            plt.plot(result["ds"], result["actual"], "b-o", label="Actual")
            plt.plot(result["ds"], result["PatchTST"], "g-s", label="Forecast")
            for _, row in anomaly_events.iterrows():
                plt.axvline(row["ds"], color="red", linestyle="--", alpha=0.6)
            plt.title(f"GC=F - 30-Day Anomaly Analysis | RMSE={rmse:.3f} | Model Error Threshold=3%")
            plt.xlabel("Date"); plt.ylabel("Price"); plt.legend()
            plt.tight_layout(); plt.show()

            # ─── LLM: her anomali olayı için tek seferlik ───
            for _, row in anomaly_events.iterrows():
                news = get_news(row["ds"])
                rag  = get_rag_context(
                    f"gold price movement explanation: "
                    f"federal reserve interest rate impact, "
                    f"geopolitical risk safe haven demand, "
                    f"inflation deflation gold correlation, "
                    f"dollar strength weakness gold inverse relationship, "
                    f"technical resistance support breakout, "
                    f"actual price: ${row['actual']:.0f}, "
                    f"deviation: {row['deviation_pct']:.1f}%"
                )
                prompt = f"""
                Gold price anomaly detected on {row['ds'].date()}.
                Actual price: ${row['actual']:.0f}
                5-day rolling average: ${row['rolling_mean']:.0f}
                Model prediction error: {row['error_pct']:.1f}% (${row['error']:.0f})

                Related news that day:
                {news}

                Relevant theory from technical analysis books:
                {rag}

                Based on the news and theory, briefly explain why this anomaly may have occurred.
                """
                print(f"\n***** {row['ds'].date()} — Model Error: {row['error_pct']:.1f}% (${row['error']:.0f})")
                print(f"News: {news}")
                print(f"Explanation: {ask_llm(prompt)}")

        # ─────────────────────────────────────
        elif choice == "3":
            print("\n─── HOLDOUT BACKTEST ───")
            offset_back  = int(input("Go back from LAST by how many points? (e.g. 500): "))
            train_points = int(input("How many points BEFORE that to train on? (e.g. 3000): "))
            horizon      = int(input("Forecast horizon points AFTER split? (e.g. 90): "))
            input_size   = int(input("Input (lag window) size? (e.g. 60): "))
            max_steps    = 50
            n_heads      = int(input("n_heads? (e.g. 4): "))
            patch_len    = int(input("patch_len? (e.g. 16): "))

            split_idx = len(df) - offset_back
            train_df  = df.iloc[split_idx - train_points:split_idx].reset_index(drop=True)
            actual_df = df.iloc[split_idx:split_idx + horizon].reset_index(drop=True)

            model = NeuralForecast(
                models=[PatchTST(h=horizon, input_size=input_size, max_steps=max_steps, n_heads=n_heads, patch_len=patch_len)],
                freq="B"
            )
            model.fit(train_df)
            preds = model.predict().reset_index(drop=True)
            preds["ds"]     = pd.to_datetime(preds["ds"])
            actual_df["ds"] = pd.to_datetime(actual_df["ds"])
            result = preds.merge(actual_df[["ds", "y"]], on="ds", how="inner")

            rmse = np.sqrt(((result["y"] - result["PatchTST"]) ** 2).mean())
            print(f"\nRMSE: {rmse:.3f}")

            plt.figure(figsize=(14, 5))
            plt.plot(result["ds"], result["y"], "b-o", label="Actual")
            plt.plot(result["ds"], result["PatchTST"], "g-s", label="Forecast")
            for d in result["ds"]:
                plt.axvline(d, color="red", linestyle="--", alpha=0.2)
            plt.title(f"GC=F - Holdout Backtest | RMSE={rmse:.3f}")
            plt.xlabel("Date"); plt.ylabel("Price"); plt.legend()
            plt.tight_layout(); plt.show()

    except ValueError as e:
        print(f"Hata: {e}")