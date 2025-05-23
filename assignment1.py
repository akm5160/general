import yfinance as yf
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import os, requests
from datetime import datetime, timedelta

load_dotenv()

# === Get ticker from Yahoo ===
def resolve_company_name_to_ticker(name):
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": name, "quotesCount": 1}
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, params=params, headers=headers)
    return res.json()["quotes"][0]["symbol"]

# === Get latest news from Finnhub ===
def get_news_finnhub(ticker):
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d"),
        "to": datetime.today().strftime("%Y-%m-%d"),
        "token": os.getenv("FINNHUB_API_KEY")
    }
    res = requests.get(url, params=params)
    return [f"{n['headline']} - {n['url']}" for n in res.json()[:5]]

# === GPT-4o-mini Azure Setup ===
llm = AzureChatOpenAI(
    deployment_name="gpt4o-mini",
    openai_api_version="2024-03-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

# === Analysis Function ===
def analyze_news_with_gpt(news_items, company, ticker, llm):
    news_text = "\n\n".join(news_items)
    prompt = f"""
You are an AI financial analyst.

Analyze the following news headlines about {company} (Ticker: {ticker}). Extract:

- Sentiment (positive, negative, neutral)
- Mentioned people
- Mentioned places
- Related industries
- Market implications
- Confidence score (0 to 1)

Respond in JSON format with keys:
["company_name", "stock_code", "news_desc", "sentiment", "peoplenames", "places_names", "related_industries", "market_implications", "confidence_score"]

News:
{news_text}
"""
    response = llm.invoke(prompt)
    return response.content

# === RUN ===
company = "Tesla Inc"
ticker = resolve_company_name_to_ticker(company)
news = get_news_finnhub(ticker)
structured_json = analyze_news_with_gpt(news, company, ticker, llm)

print("\n=== JSON Output ===\n", structured_json)




#####SAMPLE OUTPUT#####
#######################
# /Users/macbook/Documents/assignment1/az.py:58: LangChainDeprecationWarning: The class `AzureChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import AzureChatOpenAI``.
#   llm = AzureChatOpenAI(
# /Users/macbook/Documents/assignment1/az.py:67: LangChainDeprecationWarning: LangChain agents will continue to be supported, but it is recommended for new use cases to be built with LangGraph. LangGraph offers a more flexible and full-featured framework for building agents, including support for tool-calling, persistence of state, and human-in-the-loop workflows. For details, refer to the `LangGraph documentation <https://langchain-ai.github.io/langgraph/>`_ as well as guides for `Migrating from AgentExecutor <https://python.langchain.com/docs/how_to/migrate_agent/>`_ and LangGraph's `Pre-built ReAct agent <https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/>`_.
#   agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# /Users/macbook/Documents/assignment1/az.py:70: LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
#   response = agent.run("Give me the last 3 closing prices and news for Nvidia")


# > Entering new AgentExecutor chain...
# I need to find the ticker symbol for Nvidia first in order to retrieve the last 3 closing prices and the latest news.  
# Action: ResolveTicker  
# Action Input: "Nvidia"  
# Observation: NVDA
# Thought:Now that I have the ticker symbol for Nvidia (NVDA), I can retrieve the last 3 closing prices.  
# Action: Last3ClosingPrices  
# Action Input: "NVDA"  
# Observation: 2025-05-20: 134.38
# 2025-05-21: 131.80
# 2025-05-22: 132.83
# Thought:I have obtained the last 3 closing prices for Nvidia. Now, I will retrieve the latest news for the same ticker.  
# Action: LatestNews  
# Action Input: "NVDA"  
# Observation: Intel Doesn't Need To Beat TSMC To Win - Nearing $20 Buy Signal - https://finnhub.io/api/news?id=2f26159412f55730d7dc5bcdafbc472ee3781820c18496c2b93b458e4b37a22a
# Kraken Tokenizes Apple, Nvidia, Tesla Shares - https://finnhub.io/api/news?id=83fd210df26e6401cbc7c81f6cb510492125dd8c9a653f150d382222369da6c0
# Did You Survive The Great Crash Of 2025? - https://finnhub.io/api/news?id=13194666da901c7b1804ea3dcb1294154b7bf74746ef6f1efeacc3cc3866fddb
# AMD: Positives Everywhere - https://finnhub.io/api/news?id=633e24862ab8a5f88a1bfc6bebdf25c55ddace6eb92773d63c7118109e31d48c
# 3 No-Brainer Artificial Intelligence (AI) Stocks to Buy Right Now - https://finnhub.io/api/news?id=ba394368420e7729b30db20d1c7cc9e48f60f3b65164fdbd5f3ebade13d321ee
# Thought:I have gathered the last 3 closing prices and the latest news articles related to Nvidia. 

# Final Answer: 
# Last 3 closing prices for Nvidia (NVDA):
# - 2025-05-20: $134.38
# - 2025-05-21: $131.80
# - 2025-05-22: $132.83

# Latest news articles:
# 1. [Intel Doesn't Need To Beat TSMC To Win - Nearing $20 Buy Signal](https://finnhub.io/api/news?id=2f26159412f55730d7dc5bcdafbc472ee3781820c18496c2b93b458e4b37a22a)
# 2. [Kraken Tokenizes Apple, Nvidia, Tesla Shares](https://finnhub.io/api/news?id=83fd210df26e6401cbc7c81f6cb510492125dd8c9a653f150d382222369da6c0)
# 3. [Did You Survive The Great Crash Of 2025?](https://finnhub.io/api/news?id=13194666da901c7b1804ea3dcb1294154b7bf74746ef6f1efeacc3cc3866fddb)
# 4. [AMD: Positives Everywhere](https://finnhub.io/api/news?id=633e24862ab8a5f88a1bfc6bebdf25c55ddace6eb92773d63c7118109e31d48c)
# 5. [3 No-Brainer Artificial Intelligence (AI) Stocks to Buy Right Now](https://finnhub.io/api/news?id=ba394368420e7729b30db20d1c7cc9e48f60f3b65164fdbd5f3ebade13d321ee)

# > Finished chain.

# === Final Output ===
#  Last 3 closing prices for Nvidia (NVDA):
# - 2025-05-20: $134.38
# - 2025-05-21: $131.80
# - 2025-05-22: $132.83

# Latest news articles:
# 1. [Intel Doesn't Need To Beat TSMC To Win - Nearing $20 Buy Signal](https://finnhub.io/api/news?id=2f26159412f55730d7dc5bcdafbc472ee3781820c18496c2b93b458e4b37a22a)
# 2. [Kraken Tokenizes Apple, Nvidia, Tesla Shares](https://finnhub.io/api/news?id=83fd210df26e6401cbc7c81f6cb510492125dd8c9a653f150d382222369da6c0)
# 3. [Did You Survive The Great Crash Of 2025?](https://finnhub.io/api/news?id=13194666da901c7b1804ea3dcb1294154b7bf74746ef6f1efeacc3cc3866fddb)
# 4. [AMD: Positives Everywhere](https://finnhub.io/api/news?id=633e24862ab8a5f88a1bfc6bebdf25c55ddace6eb92773d63c7118109e31d48c)
# 5. [3 No-Brainer Artificial Intelligence (AI) Stocks to Buy Right Now](https://finnhub.io/api/news?id=ba394368420e7729b30db20d1c7cc9e48f60f3b65164fdbd5f3ebade13d321ee)
# (.venv) (base) macbook@macbooks-MacBook-Air assignment1 % python az.py
# /Users/macbook/Documents/assignment1/az.py:108: LangChainDeprecationWarning: The class `AzureChatOpenAI` was deprecated in LangChain 0.0.10 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import AzureChatOpenAI``.
#   llm = AzureChatOpenAI(

# === JSON Output ===
#  ```json
# [
#     {
#         "company_name": "Tesla Inc",
#         "stock_code": "TSLA",
#         "news_desc": "Tesla's Unboxed Manufacturing",
#         "sentiment": "positive",
#         "peoplenames": [],
#         "places_names": [],
#         "related_industries": ["automotive", "manufacturing"],
#         "market_implications": "Potentially positive for production efficiency and cost reduction, which could enhance profitability.",
#         "confidence_score": 0.8
#     },
#     {
#         "company_name": "Tesla Inc",
#         "stock_code": "TSLA",
#         "news_desc": "FTC probes Media Matters over Musk's X boycott claims, document shows",
#         "sentiment": "negative",
#         "peoplenames": ["Musk"],
#         "places_names": [],
#         "related_industries": ["media", "technology"],
#         "market_implications": "Negative sentiment could impact Tesla's brand image and investor confidence.",
#         "confidence_score": 0.7
#     },
#     {
#         "company_name": "Tesla Inc",
#         "stock_code": "TSLA",
#         "news_desc": "Did You Survive The Great Crash Of 2025?",
#         "sentiment": "neutral",
#         "peoplenames": [],
#         "places_names": [],
#         "related_industries": ["finance", "automotive"],
#         "market_implications": "Speculative nature of the headline may not have immediate implications for Tesla's stock.",
#         "confidence_score": 0.5
#     },
#     {
#         "company_name": "Tesla Inc",
#         "stock_code": "TSLA",
#         "news_desc": "US Senate votes to block California 2035 electric vehicle rules",
#         "sentiment": "negative",
#         "peoplenames": [],
#         "places_names": ["California"],
#         "related_industries": ["automotive", "government"],
#         "market_implications": "Negative for Tesla as it may hinder the growth of the electric vehicle market in California.",
#         "confidence_score": 0.75
#     },
#     {
#         "company_name": "Tesla Inc",
#         "stock_code": "TSLA",
#         "news_desc": "China's Xiaomi to start selling YU7 in July, a rival to Tesla's Model Y",
#         "sentiment": "negative",
#         "peoplenames": [],
#         "places_names": ["China"],
#         "related_industries": ["automotive", "technology"],
#         "market_implications": "Increased competition could negatively impact Tesla's market share and sales.",
#         "confidence_score": 0.8
#     }
# ]
