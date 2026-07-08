import os
import getpass
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit as st
from dotenv import load_dotenv
import hashlib
import io
import wave
import tempfile
import threading
import time

try:
    import speech_recognition as sr
    import pyaudio
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False

# Load environment variables
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env file. Please add it.")

os.environ["GROQ_API_KEY"] = api_key

# --- Enhanced Schema for Parsed Query ---
class ComparisonQuery(BaseModel):
    stocks: List[str] = Field(description="List of stock names or ticker symbols for comparison")
    duration: str = Field(description="Time period for analysis (e.g., '1 month', '30 days')")
    indicators: List[str] = Field(
        description="List of technical indicators to use",
        default=["RSI", "MACD", "SMA", "EMA", "Bollinger"]
    )
    analysis_type: str = Field(
        description="Type of analysis: 'single', 'comparison', 'recommendation', 'risk', 'swot', 'profit_calc', 'backtest'",
        default="single"
    )
    target_profit: Optional[float] = Field(description="Target profit amount for calculations", default=None)
    risk_tolerance: Optional[str] = Field(description="Risk tolerance level: low, medium, high", default="medium")
    backtest_strategies: Optional[List[str]] = Field(description="Strategies for backtesting", default=["RSI", "MACD", "Bollinger"])
    backtest_amount: Optional[float] = Field(description="Fixed amount per trade for backtesting", default=10000)
    backtest_stocks: Optional[List[str]] = Field(description="Specific stocks for backtesting", default=["TCS", "INFY", "HDFC", "ICICI", "RELIANCE"])

parser = PydanticOutputParser(pydantic_object=ComparisonQuery)

# --- LLM Setup ---
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=2000
)

# --- Backtesting Data Models ---
class Trade:
    def __init__(self, stock, strategy, date, action, price, quantity, amount):
        self.stock = stock
        self.strategy = strategy
        self.date = date
        self.action = action  # 'BUY' or 'SELL'
        self.price = price
        self.quantity = quantity
        self.amount = amount
        self.pnl = 0.0

class BacktestResult:
    def __init__(self, stock, strategy, total_trades, winning_trades, losing_trades, total_pnl, win_rate, avg_pnl):
        self.stock = stock
        self.strategy = strategy
        self.total_trades = total_trades
        self.winning_trades = winning_trades
        self.losing_trades = losing_trades
        self.total_pnl = total_pnl
        self.win_rate = win_rate
        self.avg_pnl = avg_pnl

# --- Enhanced State Definition ---
class AnalysisState:
    def __init__(self):
        self.user_text = ""
        self.stocks = []
        self.duration = ""
        self.indicators = []
        self.analysis_type = "single"
        self.data = {}
        self.analysis_results = {}
        self.comparison_table = None
        self.target_profit = None
        self.risk_tolerance = "medium"
        self.conversation_context = []
        # Backtesting specific
        self.backtest_strategies = ["RSI", "MACD", "Bollinger"]
        self.backtest_amount = 10000
        self.backtest_stocks = ["TCS", "INFY", "HDFC", "ICICI", "RELIANCE"]
        self.backtest_results = {}
        self.backtest_trades = []

# --- Enhanced Query Parser with Better Instructions ---
def parse_user_query(user_text: str, context: List[Dict] = None) -> ComparisonQuery:
    """Parse user query with context awareness"""
    print(f"🔍 Parsing query: {user_text}")
    
    # Build context string from previous conversations
    context_str = ""
    if context and len(context) > 0:
        recent_context = context[-3:]  # Last 3 exchanges
        for msg in recent_context:
            if msg["role"] == "user":
                context_str += f"Previous user query: {msg['content'][:100]}...\n"
    
    prompt = f"""
You are a stock analysis assistant. Extract information from this trading query.
Consider the conversation context if provided.

{context_str}

Current Query: "{user_text}"

Return ONLY a JSON object with this EXACT structure:
{{
  "stocks": ["STOCK1", "STOCK2", ...],
  "duration": "TIME_PERIOD",
  "indicators": ["INDICATOR1", "INDICATOR2"],
  "analysis_type": "TYPE",
  "target_profit": null or number,
  "risk_tolerance": "low" or "medium" or "high"
}}

Analysis Types:
- "single": Analysis of one stock
- "comparison": Compare multiple stocks
- "recommendation": Which stock is better to invest
- "risk": Risk assessment of stocks
- "swot": SWOT analysis
- "profit_calc": Calculate capital needed for target profit
- "backtest": Run backtesting with multiple strategies on multiple stocks

Rules:
- Extract ALL stock symbols mentioned (common ones: TCS, INFY, HDFC, ICICI, RELIANCE, AAPL, MSFT, GOOGL, AMZN)
- If user mentions "risk assessment" or "risk analysis", set analysis_type to "risk"
- If user mentions "SWOT" or "strengths weaknesses", set analysis_type to "swot"
- If user mentions "profit" with a number or "capital needed", set analysis_type to "profit_calc"
- If user mentions "backtest", "backtesting", "strategy testing", "strategy comparison", set analysis_type to "backtest"
- If user mentions specific profit amount (e.g., "$1000", "10000 rupees"), extract it as target_profit
- If user mentions specific stocks for backtesting, extract them as backtest_stocks
- If user mentions specific amount like "10000" or "50000", set backtest_amount to that value
- Default backtest_stocks: ["TCS", "INFY", "HDFC", "ICICI", "RELIANCE"]
- Default indicators if none mentioned: ["RSI", "MACD", "SMA", "EMA", "Bollinger"]
- Default duration if none mentioned: "1 month"

Examples:
Query: "Do a risk assessment of TCS and INFY"
{{
  "stocks": ["TCS", "INFY"],
  "duration": "1 month",
  "indicators": ["RSI", "MACD", "SMA", "EMA", "Bollinger"],
  "analysis_type": "risk",
  "target_profit": null,
  "risk_tolerance": "medium"
}}

Query: "SWOT analysis of HDFC"
{{
  "stocks": ["HDFC"],
  "duration": "1 month", 
  "indicators": ["RSI", "MACD", "SMA", "EMA", "Bollinger"],
  "analysis_type": "swot",
  "target_profit": null,
  "risk_tolerance": "medium"
}}

Query: "How much capital do I need to make $5000 profit from AAPL?"
{{
  "stocks": ["AAPL"],
  "duration": "1 month",
  "indicators": ["RSI", "MACD", "SMA", "EMA", "Bollinger"],
  "analysis_type": "profit_calc",
  "target_profit": 5000,
  "risk_tolerance": "medium"
}}

Query: "Run backtest on TCS, INFY, HDFC with RSI, MACD and Bollinger strategies"
{{
  "stocks": ["TCS", "INFY", "HDFC"],
  "duration": "1 month",
  "indicators": ["RSI", "MACD", "Bollinger"],
  "analysis_type": "backtest",
  "backtest_strategies": ["RSI", "MACD", "Bollinger"],
  "backtest_amount": 10000,
  "backtest_stocks": ["TCS", "INFY", "HDFC"],
  "risk_tolerance": "medium"
}}

Query: "Backtest RELIANCE and WIPRO with 50000 amount"
{{
  "stocks": ["RELIANCE", "WIPRO"],
  "duration": "1 month",
  "indicators": ["RSI", "MACD", "Bollinger"],
  "analysis_type": "backtest",
  "backtest_strategies": ["RSI", "MACD", "Bollinger"],
  "backtest_amount": 50000,
  "backtest_stocks": ["RELIANCE", "WIPRO"],
  "risk_tolerance": "medium"
}}

Query: "Backtest TCS only with RSI strategy"
{{
  "stocks": ["TCS"],
  "duration": "1 month",
  "indicators": ["RSI"],
  "analysis_type": "backtest",
  "backtest_strategies": ["RSI"],
  "backtest_amount": 10000,
  "backtest_stocks": ["TCS"],
  "risk_tolerance": "medium"
}}

Return ONLY the JSON, no explanations.
"""

    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()

        # Clean the response
        if "```" in response_text:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

        # Parse JSON
        parsed_json = json.loads(response_text)

        # For backtesting, use the stocks from the query, otherwise use defaults
        analysis_type = parsed_json.get("analysis_type", "single")
        if analysis_type == "backtest":
            backtest_stocks = parsed_json.get("backtest_stocks", parsed_json.get("stocks", ["TCS", "INFY", "HDFC", "ICICI", "RELIANCE"]))
        else:
            backtest_stocks = ["TCS", "INFY", "HDFC", "ICICI", "RELIANCE"]
        
        result = ComparisonQuery(
            stocks=parsed_json.get("stocks", ["TCS"]),
            duration=parsed_json.get("duration", "1 month"),
            indicators=parsed_json.get("indicators", ["RSI", "MACD", "SMA", "EMA", "Bollinger"]),
            analysis_type=analysis_type,
            target_profit=parsed_json.get("target_profit"),
            risk_tolerance=parsed_json.get("risk_tolerance", "medium"),
            backtest_strategies=parsed_json.get("backtest_strategies", ["RSI", "MACD", "Bollinger"]),
            backtest_amount=parsed_json.get("backtest_amount", 10000),
            backtest_stocks=backtest_stocks
        )

    except Exception as e:
        print(f"⚠️ LLM parsing failed: {e}")
        print("🔄 Using fallback parsing...")
        result = fallback_parse_enhanced(user_text)

    print("✅ Parsed Query:")
    print(f"  Stocks: {result.stocks}")
    print(f"  Duration: {result.duration}")
    print(f"  Indicators: {result.indicators}")
    print(f"  Analysis Type: {result.analysis_type}")
    print(f"  Target Profit: {result.target_profit}")
    print(f"  Risk Tolerance: {result.risk_tolerance}")

    return result

def fallback_parse_enhanced(user_text: str) -> ComparisonQuery:
    """Enhanced fallback parsing with additional analysis types"""
    user_text_upper = user_text.upper()
    user_text_lower = user_text.lower()

    # Extract stocks
    stock_patterns = [
        r'\b([A-Z]{2,6})\b',
        r'\b(TCS|INFY|HDFC|ICICI|RELIANCE|AAPL|MSFT|GOOGL|AMZN)\b'
    ]

    stocks = []
    for pattern in stock_patterns:
        matches = re.findall(pattern, user_text_upper)
        stocks.extend(matches)

    stocks = list(set(stocks))
    if not stocks:
        stocks = ["TCS"]

    # Determine analysis type
    analysis_type = "single"
    if len(stocks) > 1:
        analysis_type = "comparison"
    
    if any(word in user_text_lower for word in ["risk", "volatility", "risk assessment"]):
        analysis_type = "risk"
    elif any(word in user_text_lower for word in ["swot", "strengths", "weaknesses", "opportunities", "threats"]):
        analysis_type = "swot"
    elif any(word in user_text_lower for word in ["capital", "profit", "money needed", "investment needed"]):
        analysis_type = "profit_calc"
    elif any(word in user_text_lower for word in ["backtest", "backtesting", "strategy testing", "strategy comparison", "test strategies"]):
        analysis_type = "backtest"
    elif any(word in user_text_lower for word in ["recommend", "suggest", "which", "better", "best"]):
        analysis_type = "recommendation"

    # Extract target profit
    target_profit = None
    profit_patterns = [r'\$(\d+)', r'(\d+)\s*(?:rupees|rs|inr|dollars|usd)']
    for pattern in profit_patterns:
        match = re.search(pattern, user_text_lower)
        if match:
            target_profit = float(match.group(1))
            break

    # Extract duration
    duration = "1 month"
    duration_patterns = [
        r'(\d+)\s*(day|days)',
        r'(\d+)\s*(week|weeks)',
        r'(\d+)\s*(month|months)',
        r'(\d+)\s*(year|years)'
    ]

    for pattern in duration_patterns:
        match = re.search(pattern, user_text_lower)
        if match:
            num = match.group(1)
            unit = match.group(2)
            if not unit.endswith('s'):
                unit += 's' if int(num) > 1 else ''
            duration = f"{num} {unit}"
            break

    # Extract indicators
    all_indicators = ["RSI", "MACD", "SMA", "EMA", "BOLLINGER"]
    mentioned_indicators = []

    for indicator in all_indicators:
        if indicator in user_text_upper:
            if indicator == "BOLLINGER":
                mentioned_indicators.append("Bollinger")
            else:
                mentioned_indicators.append(indicator)

    if not mentioned_indicators:
        mentioned_indicators = ["RSI", "MACD", "SMA", "EMA", "Bollinger"]

    # For backtesting, use the extracted stocks, otherwise use default
    if analysis_type == "backtest":
        backtest_stocks = stocks if stocks else ["TCS", "INFY", "HDFC", "ICICI", "RELIANCE"]
    else:
        backtest_stocks = ["TCS", "INFY", "HDFC", "ICICI", "RELIANCE"]
    
    return ComparisonQuery(
        stocks=stocks,
        duration=duration,
        indicators=mentioned_indicators,
        analysis_type=analysis_type,
        target_profit=target_profit,
        risk_tolerance="medium",
        backtest_strategies=["RSI", "MACD", "Bollinger"],
        backtest_amount=10000,
        backtest_stocks=backtest_stocks
    )

# --- Enhanced Data Generation with Stock-specific Characteristics ---
def generate_stock_data(stock_symbol: str, days=30, initial_price=100, minutes_per_day=390):
    """Generate synthetic stock data with unique characteristics per stock"""
    
    # Create unique characteristics based on stock symbol
    seed = int(hashlib.md5(stock_symbol.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    
    # Stock-specific parameters
    stock_profiles = {
        'TCS': {'volatility': 0.015, 'trend': 0.0002, 'base_price': 3500},
        'INFY': {'volatility': 0.018, 'trend': 0.00015, 'base_price': 1500},
        'HDFC': {'volatility': 0.020, 'trend': 0.0001, 'base_price': 1600},
        'ICICI': {'volatility': 0.022, 'trend': 0.00018, 'base_price': 950},
        'RELIANCE': {'volatility': 0.025, 'trend': -0.0001, 'base_price': 2400},
        'AAPL': {'volatility': 0.020, 'trend': 0.00025, 'base_price': 175},
        'MSFT': {'volatility': 0.018, 'trend': 0.00022, 'base_price': 380},
        'GOOGL': {'volatility': 0.022, 'trend': 0.00020, 'base_price': 140},
        'AMZN': {'volatility': 0.024, 'trend': 0.00018, 'base_price': 155}
    }
    
    # Get stock-specific parameters or use defaults
    if stock_symbol in stock_profiles:
        profile = stock_profiles[stock_symbol]
        volatility = profile['volatility']
        trend = profile['trend']
        initial_price = profile['base_price']
    else:
        volatility = 0.02 + np.random.uniform(-0.01, 0.01)
        trend = np.random.uniform(-0.0002, 0.0003)
        initial_price = 100 + np.random.uniform(0, 500)
    
    total_minutes = days * minutes_per_day
    start_date = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0) - timedelta(days=days)
    timestamps = []
    
    for day in range(days):
        day_start = start_date + timedelta(days=day)
        for minute in range(minutes_per_day):
            timestamps.append(day_start + timedelta(minutes=minute))
    
    # Generate unique price movements
    returns = np.random.normal(trend, volatility, total_minutes)
    
    # Add stock-specific patterns
    for i in range(len(returns)):
        minute_of_day = i % minutes_per_day
        
        # Opening and closing volatility
        if minute_of_day < 30 or minute_of_day > 360:
            returns[i] *= 1.5
        
        # Add some mean reversion
        if i > 0 and abs(returns[i-1]) > 2 * volatility:
            returns[i] -= returns[i-1] * 0.3
        
        # Add weekly patterns (some stocks stronger on certain days)
        day_of_week = (i // minutes_per_day) % 5
        if stock_symbol in ['TCS', 'INFY'] and day_of_week == 0:  # Tech stocks stronger on Mondays
            returns[i] += 0.0001
        elif stock_symbol in ['HDFC', 'ICICI'] and day_of_week == 4:  # Banking stocks on Fridays
            returns[i] += 0.00008
    
    # Generate prices
    prices = [initial_price]
    for i in range(1, total_minutes):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, initial_price * 0.5))  # Prevent going too low
    
    # Generate volume with stock-specific patterns
    base_volume = 5000 * (1 + seed % 10)
    volumes = np.random.poisson(base_volume, total_minutes)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'volume': volumes
    })
    df['date'] = df['timestamp'].dt.date
    
    return df

# --- Backtesting Strategy Functions ---
def rsi_strategy(df, current_idx, rsi_threshold_low=30, rsi_threshold_high=70):
    """RSI Strategy: Buy when RSI < 30, Sell when RSI > 70"""
    if current_idx < 14:  # Need at least 14 periods for RSI
        return None
    
    current_rsi = df['RSI'].iloc[current_idx]
    if pd.isna(current_rsi):
        return None
    
    if current_rsi < rsi_threshold_low:
        return 'BUY'
    elif current_rsi > rsi_threshold_high:
        return 'SELL'
    return None

def macd_strategy(df, current_idx):
    """MACD Strategy: Buy when MACD line crosses above signal, Sell when below"""
    if current_idx < 26:  # Need at least 26 periods for MACD
        return None
    
    if 'MACD_line' not in df.columns or 'MACD_signal' not in df.columns:
        return None
    
    current_macd = df['MACD_line'].iloc[current_idx]
    current_signal = df['MACD_signal'].iloc[current_idx]
    prev_macd = df['MACD_line'].iloc[current_idx - 1]
    prev_signal = df['MACD_signal'].iloc[current_idx - 1]
    
    if pd.isna(current_macd) or pd.isna(current_signal) or pd.isna(prev_macd) or pd.isna(prev_signal):
        return None
    
    # Golden cross: MACD crosses above signal
    if prev_macd <= prev_signal and current_macd > current_signal:
        return 'BUY'
    # Death cross: MACD crosses below signal
    elif prev_macd >= prev_signal and current_macd < current_signal:
        return 'SELL'
    return None

def bollinger_strategy(df, current_idx):
    """Bollinger Bands Strategy: Buy when price touches lower band, Sell when touches upper band"""
    if current_idx < 20:  # Need at least 20 periods for Bollinger Bands
        return None
    
    if 'BB_upper' not in df.columns or 'BB_lower' not in df.columns:
        return None
    
    current_price = df['price'].iloc[current_idx]
    current_upper = df['BB_upper'].iloc[current_idx]
    current_lower = df['BB_lower'].iloc[current_idx]
    
    if pd.isna(current_price) or pd.isna(current_upper) or pd.isna(current_lower):
        return None
    
    # Price touches or goes below lower band
    if current_price <= current_lower:
        return 'BUY'
    # Price touches or goes above upper band
    elif current_price >= current_upper:
        return 'SELL'
    return None

def extract_days_from_duration(duration_str: str) -> int:
    """Convert human-readable duration to days"""
    duration_str = duration_str.lower()
    numbers = re.findall(r'\d+', duration_str)
    if not numbers:
        return 30

    num = int(numbers[0])

    if 'year' in duration_str:
        return num * 365
    elif 'month' in duration_str:
        return num * 30
    elif 'week' in duration_str:
        return num * 7
    elif 'day' in duration_str:
        return num
    else:
        return 30

# --- Technical Indicator Functions ---
def calculate_indicators(df: pd.DataFrame, indicators: List[str]):
    """Calculate selected technical indicators"""
    df = df.copy()

    # RSI
    if 'RSI' in indicators:
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

    # SMA
    if 'SMA' in indicators:
        df['SMA_20'] = df['price'].rolling(window=20).mean()
        df['SMA_50'] = df['price'].rolling(window=50).mean()

    # EMA
    if 'EMA' in indicators:
        df['EMA_20'] = df['price'].ewm(span=20).mean()
        df['EMA_50'] = df['price'].ewm(span=50).mean()

    # Bollinger Bands
    if 'Bollinger' in indicators:
        df['BB_mid'] = df['price'].rolling(window=20).mean()
        bb_std = df['price'].rolling(window=20).std()
        df['BB_upper'] = df['BB_mid'] + (bb_std * 2)
        df['BB_lower'] = df['BB_mid'] - (bb_std * 2)

    # MACD
    if 'MACD' in indicators:
        ema12 = df['price'].ewm(span=12).mean()
        ema26 = df['price'].ewm(span=26).mean()
        df['MACD_line'] = ema12 - ema26
        df['MACD_signal'] = df['MACD_line'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']

    return df

# --- Risk Assessment Function ---
def perform_risk_assessment(state: AnalysisState) -> Dict[str, Any]:
    """Perform comprehensive risk assessment"""
    risk_data = {}
    
    for stock, df in state.data.items():
        prices = df['price'].values
        returns = df['price'].pct_change().dropna()
        
        # Calculate risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (returns.mean() * 252) / (volatility + 0.0001)
        max_drawdown = ((df['price'].cummax() - df['price']) / df['price'].cummax()).max()
        var_95 = returns.quantile(0.05)  # Value at Risk (95% confidence)
        
        # Beta calculation (simplified - against market average)
        if len(state.stocks) > 1:
            market_returns = pd.concat([state.data[s]['price'].pct_change() for s in state.stocks]).mean()
            beta = returns.cov(market_returns) / market_returns.var() if market_returns.var() > 0 else 1.0
        else:
            beta = 1.0
        
        risk_data[stock] = {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'beta': beta,
            'risk_score': calculate_risk_score(volatility, sharpe_ratio, max_drawdown)
        }
    
    return risk_data

def calculate_risk_score(volatility, sharpe_ratio, max_drawdown):
    """Calculate overall risk score (0-100, lower is better)"""
    vol_score = min(volatility * 100, 50)  # Max 50 points
    sharpe_score = max(0, 25 - sharpe_ratio * 5)  # Max 25 points
    dd_score = min(max_drawdown * 100, 25)  # Max 25 points
    
    return vol_score + sharpe_score + dd_score

# --- SWOT Analysis Function ---
def perform_swot_analysis(state: AnalysisState) -> Dict[str, Dict]:
    """Generate SWOT analysis based on technical indicators"""
    swot_results = {}
    
    for stock, df in state.data.items():
        current_price = df['price'].iloc[-1]
        price_change = ((current_price - df['price'].iloc[0]) / df['price'].iloc[0]) * 100
        
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []
        
        # Analyze based on indicators
        if 'RSI' in df.columns:
            rsi_current = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            if 30 <= rsi_current <= 70:
                strengths.append(f"RSI in healthy range ({rsi_current:.1f})")
            elif rsi_current < 30:
                opportunities.append(f"Oversold condition (RSI: {rsi_current:.1f})")
            else:
                weaknesses.append(f"Overbought condition (RSI: {rsi_current:.1f})")
        
        if 'MACD_line' in df.columns:
            macd_current = df['MACD_line'].iloc[-1]
            macd_signal = df['MACD_signal'].iloc[-1]
            if not pd.isna(macd_current) and not pd.isna(macd_signal):
                if macd_current > macd_signal:
                    strengths.append("Positive MACD crossover")
                else:
                    weaknesses.append("Negative MACD crossover")
        
        # Price trend analysis
        if price_change > 5:
            strengths.append(f"Strong upward trend (+{price_change:.1f}%)")
        elif price_change < -5:
            threats.append(f"Downward trend ({price_change:.1f}%)")
        
        # Volume analysis
        avg_volume = df['volume'].mean()
        recent_volume = df['volume'].tail(20).mean()
        if recent_volume > avg_volume * 1.2:
            opportunities.append("Increasing trading volume")
        elif recent_volume < avg_volume * 0.8:
            threats.append("Declining trading interest")
        
        # Bollinger Bands analysis
        if 'BB_upper' in df.columns:
            bb_position = (current_price - df['BB_lower'].iloc[-1]) / (df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1])
            if 0.3 <= bb_position <= 0.7:
                strengths.append("Price well-positioned in Bollinger Bands")
            elif bb_position > 0.9:
                threats.append("Price near upper Bollinger Band")
        
        swot_results[stock] = {
            'strengths': strengths if strengths else ["Stable price movement"],
            'weaknesses': weaknesses if weaknesses else ["No significant weaknesses identified"],
            'opportunities': opportunities if opportunities else ["Potential for growth"],
            'threats': threats if threats else ["Market volatility risk"]
        }
    
    return swot_results

# --- Profit Calculation Function ---
def calculate_capital_for_profit(state: AnalysisState) -> Dict[str, Any]:
    """Calculate capital needed for target profit"""
    if not state.target_profit:
        return {"error": "No target profit specified"}
    
    results = {}
    for stock, df in state.data.items():
        current_price = df['price'].iloc[-1]
        
        # Calculate average daily return
        returns = df['price'].pct_change().dropna()
        avg_return = returns.mean()
        win_rate = (returns > 0).mean()
        
        # Conservative estimate based on historical performance
        expected_return_per_trade = avg_return * win_rate
        
        if expected_return_per_trade > 0:
            # Calculate required capital
            trades_needed = state.target_profit / (current_price * expected_return_per_trade)
            capital_needed = current_price * trades_needed
            
            results[stock] = {
                'current_price': current_price,
                'target_profit': state.target_profit,
                'capital_needed': capital_needed,
                'expected_return_per_trade': expected_return_per_trade * 100,
                'win_rate': win_rate * 100,
                'estimated_trades': int(trades_needed),
                'risk_level': assess_risk_level(returns.std())
            }
        else:
            results[stock] = {
                'current_price': current_price,
                'target_profit': state.target_profit,
                'capital_needed': "Not recommended",
                'expected_return_per_trade': expected_return_per_trade * 100,
                'win_rate': win_rate * 100,
                'note': "Negative expected returns"
            }
    
    return results

def assess_risk_level(volatility):
    """Assess risk level based on volatility"""
    if volatility < 0.01:
        return "Low"
    elif volatility < 0.025:
        return "Medium"
    else:
        return "High"

# --- Backtesting Engine ---
def run_backtest(state: AnalysisState) -> Dict[str, Any]:
    """Run backtesting on specific stocks with multiple strategies"""
    print(f"🚀 Starting backtest with {len(state.backtest_stocks)} stocks and {len(state.backtest_strategies)} strategies")
    
    # Use the specific stocks provided
    stock_symbols = state.backtest_stocks
    
    # Strategy mapping
    strategy_functions = {
        'RSI': rsi_strategy,
        'MACD': macd_strategy,
        'Bollinger': bollinger_strategy
    }
    
    all_results = []
    all_trades = []
    
    # Generate data for all stocks
    days = extract_days_from_duration(state.duration)
    stock_data = {}
    
    for stock in stock_symbols:
        print(f"📊 Generating data for {stock}...")
        df = generate_stock_data(stock, days=days)
        df = calculate_indicators(df, state.backtest_strategies)
        stock_data[stock] = df
    
    # Run backtest for each stock and strategy combination
    for stock in stock_symbols:
        df = stock_data[stock]
        
        for strategy_name in state.backtest_strategies:
            if strategy_name not in strategy_functions:
                continue
                
            strategy_func = strategy_functions[strategy_name]
            trades = []
            position = None  # None, 'BUY', or 'SELL'
            entry_price = 0
            entry_date = None
            
            # Simulate trading day by day
            for idx in range(len(df)):
                current_date = df['timestamp'].iloc[idx].date()
                current_price = df['price'].iloc[idx]
                
                # Get strategy signal
                signal = strategy_func(df, idx)
                
                # Execute trades based on signal
                if signal == 'BUY' and position != 'BUY':
                    if position == 'SELL':  # Close short position first
                        pnl = entry_price - current_price  # Profit from short
                        trade = Trade(stock, strategy_name, entry_date, 'SELL', entry_price, 
                                    state.backtest_amount / entry_price, state.backtest_amount)
                        trade.pnl = pnl
                        trades.append(trade)
                    
                    # Open long position
                    position = 'BUY'
                    entry_price = current_price
                    entry_date = current_date
                    
                elif signal == 'SELL' and position != 'SELL':
                    if position == 'BUY':  # Close long position first
                        pnl = current_price - entry_price  # Profit from long
                        trade = Trade(stock, strategy_name, entry_date, 'BUY', entry_price, 
                                    state.backtest_amount / entry_price, state.backtest_amount)
                        trade.pnl = pnl
                        trades.append(trade)
                    
                    # Open short position
                    position = 'SELL'
                    entry_price = current_price
                    entry_date = current_date
            
            # Close any remaining position at the end
            if position == 'BUY':
                final_price = df['price'].iloc[-1]
                pnl = final_price - entry_price
                trade = Trade(stock, strategy_name, entry_date, 'BUY', entry_price, 
                            state.backtest_amount / entry_price, state.backtest_amount)
                trade.pnl = pnl
                trades.append(trade)
            elif position == 'SELL':
                final_price = df['price'].iloc[-1]
                pnl = entry_price - final_price
                trade = Trade(stock, strategy_name, entry_date, 'SELL', entry_price, 
                            state.backtest_amount / entry_price, state.backtest_amount)
                trade.pnl = pnl
                trades.append(trade)
            
            # Calculate results for this stock-strategy combination
            if trades:
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t.pnl > 0])
                losing_trades = len([t for t in trades if t.pnl < 0])
                total_pnl = sum(t.pnl for t in trades)
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
                
                result = BacktestResult(stock, strategy_name, total_trades, winning_trades, 
                                      losing_trades, total_pnl, win_rate, avg_pnl)
                all_results.append(result)
                all_trades.extend(trades)
    
    # Store results in state
    state.backtest_results = all_results
    state.backtest_trades = all_trades
    
    return {
        'results': all_results,
        'trades': all_trades,
        'total_stocks': len(stock_symbols),
        'total_strategies': len(state.backtest_strategies)
    }


# --- Enhanced Comparison Table ---
def generate_comparison_table(state: AnalysisState) -> pd.DataFrame:
    """Create a comprehensive comparison table"""
    comparison_data = []

    for stock in state.stocks:
        if stock in state.data:
            df = state.data[stock]
            row = {"Stock": stock}

            # Price metrics
            row["Current_Price"] = df['price'].iloc[-1]
            row["Price_Change_%"] = ((df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]) * 100
            row["High"] = df['price'].max()
            row["Low"] = df['price'].min()
            
            # Volume
            row["Avg_Volume"] = df['volume'].mean()

            # Technical indicators
            if 'RSI' in state.indicators and 'RSI' in df.columns:
                row["RSI"] = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 0

            if 'MACD' in state.indicators and 'MACD_line' in df.columns:
                row["MACD"] = df['MACD_line'].iloc[-1] if not pd.isna(df['MACD_line'].iloc[-1]) else 0
                row["Signal"] = df['MACD_signal'].iloc[-1] if not pd.isna(df['MACD_signal'].iloc[-1]) else 0

            if 'SMA' in state.indicators and 'SMA_20' in df.columns:
                row["SMA_20"] = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else 0

            if 'EMA' in state.indicators and 'EMA_20' in df.columns:
                row["EMA_20"] = df['EMA_20'].iloc[-1] if not pd.isna(df['EMA_20'].iloc[-1]) else 0

            comparison_data.append(row)

    return pd.DataFrame(comparison_data)

# --- Enhanced Recommendation ---
def generate_recommendation(state: AnalysisState) -> str:
    """Generate detailed investment recommendation"""
    if state.comparison_table is None or len(state.comparison_table) < 1:
        return "Insufficient data for recommendation."

    df = state.comparison_table
    scores = {}
    detailed_analysis = {}

    for _, row in df.iterrows():
        stock = row['Stock']
        score = 0
        analysis = []

        # Price momentum (30 points)
        price_change = row.get('Price_Change_%', 0)
        if price_change > 10:
            score += 30
            analysis.append(f"Strong momentum (+{price_change:.1f}%)")
        elif price_change > 5:
            score += 20
            analysis.append(f"Positive momentum (+{price_change:.1f}%)")
        elif price_change > 0:
            score += 10
            analysis.append(f"Slight uptrend (+{price_change:.1f}%)")
        else:
            analysis.append(f"Negative trend ({price_change:.1f}%)")

        # RSI (25 points)
        if 'RSI' in row:
            rsi = row['RSI']
            if 40 <= rsi <= 60:
                score += 25
                analysis.append(f"Ideal RSI range ({rsi:.1f})")
            elif 30 <= rsi <= 70:
                score += 15
                analysis.append(f"Healthy RSI ({rsi:.1f})")
            elif rsi < 30:
                score += 10
                analysis.append(f"Oversold - potential bounce ({rsi:.1f})")
            else:
                analysis.append(f"Overbought - caution ({rsi:.1f})")

        # MACD (25 points)
        if 'MACD' in row and 'Signal' in row:
            if row['MACD'] > row['Signal']:
                score += 25
                analysis.append("Bullish MACD crossover")
            else:
                score += 5
                analysis.append("Bearish MACD signal")

        # Moving averages (20 points)
        current_price = row['Current_Price']
        if 'SMA_20' in row and row['SMA_20'] > 0:
            if current_price > row['SMA_20']:
                score += 10
                analysis.append("Price above SMA")
        if 'EMA_20' in row and row['EMA_20'] > 0:
            if current_price > row['EMA_20']:
                score += 10
                analysis.append("Price above EMA")

        scores[stock] = score
        detailed_analysis[stock] = analysis

    # Sort stocks by score
    sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Generate recommendation
    recommendation = "📊 **Investment Recommendation Analysis**\n\n"
    
    if sorted_stocks[0][1] >= 70:
        recommendation += f"🏆 **Strong Buy: {sorted_stocks[0][0]}** (Score: {sorted_stocks[0][1]}/100)\n\n"
    elif sorted_stocks[0][1] >= 50:
        recommendation += f"✅ **Recommended: {sorted_stocks[0][0]}** (Score: {sorted_stocks[0][1]}/100)\n\n"
    else:
        recommendation += f"⚠️ **Cautious Buy: {sorted_stocks[0][0]}** (Score: {sorted_stocks[0][1]}/100)\n\n"
    
    # Detailed analysis for top stocks
    recommendation += "**Detailed Analysis:**\n\n"
    for stock, score in sorted_stocks[:3]:
        recommendation += f"**{stock}** (Score: {score}/100)\n"
        for point in detailed_analysis[stock]:
            recommendation += f"  • {point}\n"
        recommendation += "\n"
    
    # Risk assessment
    recommendation += "**Risk Assessment:**\n"
    if sorted_stocks[0][1] >= 70:
        recommendation += "• Low-Medium Risk: Strong technical indicators suggest favorable entry\n"
    elif sorted_stocks[0][1] >= 50:
        recommendation += "• Medium Risk: Mixed signals, consider position sizing\n"
    else:
        recommendation += "• Medium-High Risk: Weak signals, consider waiting for better entry\n"
    
    return recommendation

# --- Enhanced Visualization Functions ---
def visualize_combined_indicators(state: AnalysisState):
    """Create enhanced visualizations for multiple stocks"""
    if not state.data:
        st.error("❌ No data to visualize.")
        return

    # Determine layout based on analysis type
    if state.analysis_type in ['single', 'comparison']:
        create_technical_charts(state)
    elif state.analysis_type == 'risk':
        create_risk_charts(state)
    elif state.analysis_type == 'swot':
        display_swot_analysis(state)
    elif state.analysis_type == 'profit_calc':
        display_profit_analysis(state)
    elif state.analysis_type == 'backtest':
        display_backtest_results(state)

def create_technical_charts(state: AnalysisState):
    """Create technical analysis charts"""
    # Price comparison chart
    fig_price = go.Figure()
    
    for stock, df in state.data.items():
        fig_price.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['price'],
            mode='lines',
            name=f'{stock} Price',
            line=dict(width=2)
        ))
    
    fig_price.update_layout(
        title="Price Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Individual indicator charts
    num_indicators = len(state.indicators)
    if num_indicators > 0:
        fig = make_subplots(
            rows=num_indicators, cols=1,
            shared_xaxes=True,
            subplot_titles=state.indicators,
            vertical_spacing=0.05,
            row_heights=[1/num_indicators] * num_indicators
        )

        row = 1
        for indicator in state.indicators:
            for stock, df in state.data.items():
                if indicator == 'RSI' and 'RSI' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['RSI'], mode='lines', name=f'{stock} RSI'),
                        row=row, col=1
                    )
                    # Add RSI levels
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
                    
                elif indicator == 'MACD' and 'MACD_line' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['MACD_line'], mode='lines', name=f'{stock} MACD'),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['MACD_signal'], mode='lines', name=f'{stock} Signal', line=dict(dash='dash')),
                        row=row, col=1
                    )
                    
                elif indicator == 'SMA' and 'SMA_20' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name=f'{stock} Price', opacity=0.5),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['SMA_20'], mode='lines', name=f'{stock} SMA20'),
                        row=row, col=1
                    )
                    
                elif indicator == 'EMA' and 'EMA_20' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name=f'{stock} Price', opacity=0.5),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['EMA_20'], mode='lines', name=f'{stock} EMA20'),
                        row=row, col=1
                    )
                    
                elif indicator == 'Bollinger' and 'BB_upper' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['price'], mode='lines', name=f'{stock} Price'),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['BB_upper'], mode='lines', name=f'{stock} Upper', line=dict(dash='dash')),
                        row=row, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df['timestamp'], y=df['BB_lower'], mode='lines', name=f'{stock} Lower', line=dict(dash='dash')),
                        row=row, col=1
                    )

            row += 1

        fig.update_layout(height=250 * num_indicators, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

def create_risk_charts(state: AnalysisState):
    """Create risk assessment visualizations"""
    risk_data = perform_risk_assessment(state)
    
    # Create risk metrics comparison
    stocks = list(risk_data.keys())
    metrics = ['volatility', 'sharpe_ratio', 'max_drawdown', 'risk_score']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Volatility (Annual)', 'Sharpe Ratio', 'Max Drawdown', 'Overall Risk Score'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Volatility
    fig.add_trace(
        go.Bar(x=stocks, y=[risk_data[s]['volatility'] for s in stocks], name='Volatility'),
        row=1, col=1
    )
    
    # Sharpe Ratio
    fig.add_trace(
        go.Bar(x=stocks, y=[risk_data[s]['sharpe_ratio'] for s in stocks], name='Sharpe Ratio'),
        row=1, col=2
    )
    
    # Max Drawdown
    fig.add_trace(
        go.Bar(x=stocks, y=[risk_data[s]['max_drawdown'] for s in stocks], name='Max Drawdown'),
        row=2, col=1
    )
    
    # Risk Score
    fig.add_trace(
        go.Bar(x=stocks, y=[risk_data[s]['risk_score'] for s in stocks], name='Risk Score'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display risk summary
    st.subheader("Risk Assessment Summary")
    risk_df = pd.DataFrame(risk_data).T
    risk_df = risk_df.round(3)
    st.dataframe(risk_df, use_container_width=True)

def display_swot_analysis(state: AnalysisState):
    """Display SWOT analysis in a structured format"""
    swot_results = perform_swot_analysis(state)
    
    for stock, swot in swot_results.items():
        st.subheader(f"SWOT Analysis: {stock}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**💪 Strengths:**")
            for strength in swot['strengths']:
                st.markdown(f"• {strength}")
            
            st.markdown("**⚠️ Weaknesses:**")
            for weakness in swot['weaknesses']:
                st.markdown(f"• {weakness}")
        
        with col2:
            st.markdown("**🎯 Opportunities:**")
            for opportunity in swot['opportunities']:
                st.markdown(f"• {opportunity}")
            
            st.markdown("**⛔ Threats:**")
            for threat in swot['threats']:
                st.markdown(f"• {threat}")
        
        st.divider()

def display_profit_analysis(state: AnalysisState):
    """Display profit calculation analysis"""
    profit_results = calculate_capital_for_profit(state)
    
    st.subheader(f"Capital Requirements for ${state.target_profit} Profit")
    
    for stock, data in profit_results.items():
        if 'error' not in data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Stock", stock)
                st.metric("Current Price", f"${data['current_price']:.2f}")
            
            with col2:
                if data['capital_needed'] != "Not recommended":
                    st.metric("Capital Needed", f"${data['capital_needed']:.2f}")
                    st.metric("Win Rate", f"{data['win_rate']:.1f}%")
                else:
                    st.metric("Capital Needed", data['capital_needed'])
                    st.metric("Note", data.get('note', ''))
            
            with col3:
                st.metric("Expected Return/Trade", f"{data['expected_return_per_trade']:.3f}%")
                if 'risk_level' in data:
                    st.metric("Risk Level", data['risk_level'])
            
            st.divider()

def display_backtest_results(state: AnalysisState):
    """Display comprehensive backtesting results"""
    if not state.backtest_results:
        st.error("No backtesting results available.")
        return
    
    st.subheader("📊 Backtesting Results Summary")
    
    # Overall statistics
    total_trades = sum(r.total_trades for r in state.backtest_results)
    total_pnl = sum(r.total_pnl for r in state.backtest_results)
    avg_win_rate = sum(r.win_rate for r in state.backtest_results) / len(state.backtest_results)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Trades", f"{total_trades:,}")
    with col2:
        st.metric("Total P&L", f"₹{total_pnl:,.2f}")
    with col3:
        st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
    with col4:
        st.metric("Stocks Tested", f"{len(state.backtest_stocks)}")
    
    # Strategy comparison table
    st.subheader("📈 Strategy Performance Comparison")
    
    # Group results by strategy
    strategy_results = {}
    for result in state.backtest_results:
        if result.strategy not in strategy_results:
            strategy_results[result.strategy] = []
        strategy_results[result.strategy].append(result)
    
    # Create strategy comparison table
    strategy_comparison = []
    for strategy, results in strategy_results.items():
        total_trades = sum(r.total_trades for r in results)
        total_pnl = sum(r.total_pnl for r in results)
        avg_win_rate = sum(r.win_rate for r in results) / len(results)
        avg_pnl = sum(r.avg_pnl for r in results) / len(results)
        
        strategy_comparison.append({
            'Strategy': strategy,
            'Total Trades': total_trades,
            'Total P&L': f"₹{total_pnl:,.2f}",
            'Avg Win Rate': f"{avg_win_rate:.1f}%",
            'Avg P&L/Trade': f"₹{avg_pnl:.2f}",
            'Stocks Tested': len(results)
        })
    
    strategy_df = pd.DataFrame(strategy_comparison)
    st.dataframe(strategy_df, use_container_width=True)
    
    # Top performing stocks
    st.subheader("🏆 Top Performing Stock-Strategy Combinations")
    
    # Sort by total P&L
    top_performers = sorted(state.backtest_results, key=lambda x: x.total_pnl, reverse=True)[:20]
    
    top_data = []
    for result in top_performers:
        top_data.append({
            'Stock': result.stock,
            'Strategy': result.strategy,
            'Total Trades': result.total_trades,
            'Win Rate': f"{result.win_rate:.1f}%",
            'Total P&L': f"₹{result.total_pnl:.2f}",
            'Avg P&L/Trade': f"₹{result.avg_pnl:.2f}"
        })
    
    top_df = pd.DataFrame(top_data)
    st.dataframe(top_df, use_container_width=True)
    
    # Daily P&L chart
    st.subheader("📅 Daily P&L Trend")
    
    # Group trades by date
    daily_pnl = {}
    for trade in state.backtest_trades:
        date_str = trade.date.strftime('%Y-%m-%d')
        if date_str not in daily_pnl:
            daily_pnl[date_str] = 0
        daily_pnl[date_str] += trade.pnl
    
    # Create daily P&L chart
    dates = sorted(daily_pnl.keys())
    pnl_values = [daily_pnl[date] for date in dates]
    cumulative_pnl = np.cumsum(pnl_values)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=cumulative_pnl,
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='green' if cumulative_pnl[-1] > 0 else 'red', width=2)
    ))
    
    fig.update_layout(
        title="Cumulative P&L Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L (₹)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy performance pie chart
    st.subheader("🥧 Strategy Performance Distribution")
    
    strategy_pnl = {}
    for result in state.backtest_results:
        if result.strategy not in strategy_pnl:
            strategy_pnl[result.strategy] = 0
        strategy_pnl[result.strategy] += result.total_pnl
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(strategy_pnl.keys()),
        values=list(strategy_pnl.values()),
        hole=0.3
    )])
    
    fig_pie.update_layout(
        title="P&L Distribution by Strategy",
        height=400
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Voice Input Functions ---

def setup_speech_recognizer():
    """Setup speech recognizer"""
    if not VOICE_ENABLED:
        return None

    recognizer = sr.Recognizer()

    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.8

    return recognizer


def record_audio(duration=5):
    """Record audio from microphone"""
    if not VOICE_ENABLED:
        st.warning("🎤 Voice input is not available on Streamlit Cloud.")
        return None

    recognizer = setup_speech_recognizer()

    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now!")

            recognizer.adjust_for_ambient_noise(source, duration=1)

            audio = recognizer.listen(
                source,
                timeout=duration,
                phrase_time_limit=duration
            )

            st.success("✅ Audio recorded!")
            return audio

    except sr.WaitTimeoutError:
        st.warning("⏰ No speech detected. Please try again.")
        return None

    except Exception as e:
        st.error(f"❌ Microphone error: {str(e)}")
        return None


def speech_to_text(audio):
    """Convert speech to text"""
    if not VOICE_ENABLED:
        return None

    recognizer = setup_speech_recognizer()

    engines = [
        ("Google", lambda: recognizer.recognize_google(audio))
    ]

    for engine_name, recognize_func in engines:
        try:
            text = recognize_func()
            st.success(f"🎯 Recognized using {engine_name}: {text}")
            return text

        except sr.UnknownValueError:
            continue

        except sr.RequestError:
            continue

    st.error("❌ Could not recognize speech")
    return None


def quick_voice_input():
    """Quick voice input"""
    if not VOICE_ENABLED:
        return None

    try:
        recognizer = setup_speech_recognizer()

        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            audio = recognizer.listen(
                source,
                timeout=3,
                phrase_time_limit=8
            )

            try:
                return recognizer.recognize_google(audio)

            except Exception:
                return None

    except Exception:
        return None


def voice_button_available():
    """Check if voice input is available"""
    return VOICE_ENABLED

# --- LLM Response Generation ---
def generate_llm_response(query: str, state: AnalysisState, context: List[Dict]) -> str:
    """Generate natural language response based on analysis"""
    
    # Build context from analysis results
    analysis_context = f"""
    Query: {query}
    Stocks analyzed: {', '.join(state.stocks)}
    Duration: {state.duration}
    Analysis type: {state.analysis_type}
    """
    
    if state.comparison_table is not None:
        analysis_context += f"\nComparison data available for {len(state.comparison_table)} stocks"
    
    prompt = f"""
    You are a professional stock analyst assistant. Based on the following analysis, provide a helpful response to the user's query.
    
    {analysis_context}
    
    Provide a concise, informative response that:
    1. Directly addresses the user's query
    2. Highlights key findings from the analysis
    3. Offers actionable insights when appropriate
    4. Maintains a professional but friendly tone
    
    Keep the response under 200 words unless more detail is specifically needed.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except:
        return "Analysis complete. Please review the charts and data above for detailed insights."

# --- Streamlit Chat Interface ---
def main():
    st.set_page_config(
        page_title="📊 AI Stock Analysis Chat",
        page_icon="📈",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 24px;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
        
        /* Sticky chat input at bottom */
        .main .block-container {
            padding-bottom: 100px;
        }
        
        /* Chat input container */
        .stChatInput {
            position: fixed !important;
            bottom: 0 !important;
            left: 0 !important;
            right: 0 !important;
            z-index: 999 !important;
            background: white !important;
            border-top: 1px solid #e0e0e0 !important;
            padding: 10px 20px !important;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1) !important;
        }
        
        /* Adjust for sidebar */
        .stChatInput {
            margin-left: 0 !important;
        }
        
        /* When sidebar is collapsed, adjust input */
        .main .block-container {
            margin-left: 0 !important;
        }
        
        /* Voice button styling - more specific targeting */
        .stChatInput + div button,
        div[data-testid="column"]:nth-child(2) button,
        button[data-testid="baseButton-secondary"] {
            background-color: #2196F3 !important;
            color: white !important;
            font-size: 20px !important;
            padding: 8px 16px !important;
            border-radius: 50% !important;
            width: 50px !important;
            height: 50px !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
            margin-top: 8px !important;
        }
        
        .stChatInput + div button:hover,
        div[data-testid="column"]:nth-child(2) button:hover,
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #1976D2 !important;
            transform: scale(1.05) !important;
        }
        
        /* Ensure proper alignment */
        .stChatInput + div {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Ensure content doesn't overlap with sticky input */
        .main .block-container {
            margin-bottom: 80px;
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .stChatInput {
                background: #1e1e1e !important;
                border-top: 1px solid #333 !important;
            }
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .stChatInput {
                padding: 8px 15px !important;
            }
            
            div[data-testid="column"]:nth-child(2) button {
                width: 45px;
                height: 45px;
                font-size: 18px;
            }
        }
        
        /* Ensure proper spacing on all devices */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        /* Auto-scroll to bottom */
        .stChatMessage {
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Smooth scrolling */
        html {
            scroll-behavior: smooth;
        }
        
        /* Sidebar adjustments */
        .css-1d391kg {
            z-index: 1000 !important;
        }
        
        /* Ensure chat input doesn't overlap with sidebar */
        @media (min-width: 768px) {
            .stChatInput {
                margin-left: 0 !important;
                width: 100% !important;
            }
        }
        
        /* Mobile adjustments */
        @media (max-width: 767px) {
            .stChatInput {
                margin-left: 0 !important;
                width: 100% !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 AI Stock Analysis Chat Assistant")
    st.markdown("Ask me anything about stock analysis, comparisons, risk assessment, and investment strategies!")
    
    # Voice input is now integrated with chat input below

    # Initialize session state for chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hello! I can help you with:\n\n• Stock comparisons (e.g., 'Compare TCS and INFY')\n• Risk assessments (e.g., 'Risk analysis of HDFC')\n• SWOT analysis (e.g., 'SWOT for RELIANCE')\n• Profit calculations (e.g., 'Capital needed for $5000 profit from AAPL')\n• **Backtesting** (e.g., 'Run backtest on TCS, INFY, HDFC with RSI, MACD, Bollinger')\n• Technical analysis with indicators\n\nWhat would you like to analyze today?"
        })
    
    if 'analysis_state' not in st.session_state:
        st.session_state.analysis_state = None

    # Chat messages container with proper spacing
    with st.container():
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display analysis results if available
                if message["role"] == "assistant" and "analysis" in message:
                    if message["analysis"].get("comparison_table") is not None:
                        st.dataframe(message["analysis"]["comparison_table"], use_container_width=True)
                    
                    if message["analysis"].get("charts"):
                        message["analysis"]["charts"]()
        
        # Add spacing at the bottom to prevent overlap with sticky input
        st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    # Chat input with voice button
    if voice_button_available():
        # Create a container for the input area
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                prompt = st.chat_input("Ask about stocks (e.g., 'Compare TCS and INFY with RSI and MACD')")
            
            with col2:
                if st.button("🎤", help="Click to speak", key="voice_btn"):
                    st.session_state.voice_clicked = True
    else:
        prompt = st.chat_input("Ask about stocks (e.g., 'Compare TCS and INFY with RSI and MACD')")
        st.info("💡 Install voice dependencies for voice input: pip install SpeechRecognition PyAudio")
    
    # Handle voice input
    if st.session_state.get('voice_clicked', False):
        st.session_state.voice_clicked = False
        with st.spinner("🎤 Listening... Speak now!"):
            voice_text = quick_voice_input()
            if voice_text:
                prompt = voice_text
                st.success(f"🎯 Heard: {voice_text}")
            else:
                st.warning("⚠️ Could not detect speech. Please try again or type your message.")
    
    # Process input
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Parse query with context
                    parsed_query = parse_user_query(prompt, st.session_state.messages)
                    
                    # Create state
                    state = AnalysisState()
                    state.user_text = prompt
                    state.stocks = parsed_query.stocks
                    state.duration = parsed_query.duration
                    state.indicators = parsed_query.indicators
                    state.analysis_type = parsed_query.analysis_type
                    state.target_profit = parsed_query.target_profit
                    state.risk_tolerance = parsed_query.risk_tolerance
                    state.backtest_strategies = parsed_query.backtest_strategies
                    state.backtest_amount = parsed_query.backtest_amount
                    state.backtest_stocks = parsed_query.backtest_stocks
                    
                    # Generate data with unique characteristics per stock
                    days = extract_days_from_duration(state.duration)
                    for stock in state.stocks:
                        state.data[stock] = generate_stock_data(stock, days=days)
                        state.data[stock] = calculate_indicators(state.data[stock], state.indicators)
                    
                    # Generate comparison table
                    state.comparison_table = generate_comparison_table(state)
                    
                    # Store state
                    st.session_state.analysis_state = state
                    
                    # Generate appropriate response based on analysis type
                    if state.analysis_type == "single":
                        response = f"📈 **Analysis for {state.stocks[0]} over {state.duration}**\n\n"
                        response += "I've analyzed the stock with the requested indicators. Here are the key metrics:"
                        st.markdown(response)
                        st.dataframe(state.comparison_table, use_container_width=True)
                        visualize_combined_indicators(state)
                        
                    elif state.analysis_type == "comparison":
                        response = f"📊 **Comparison of {', '.join(state.stocks)} over {state.duration}**\n\n"
                        response += "Here's a detailed comparison of the stocks:"
                        st.markdown(response)
                        st.dataframe(state.comparison_table, use_container_width=True)
                        visualize_combined_indicators(state)
                        
                    elif state.analysis_type == "recommendation":
                        response = generate_recommendation(state)
                        st.markdown(response)
                        st.dataframe(state.comparison_table, use_container_width=True)
                        visualize_combined_indicators(state)
                        
                    elif state.analysis_type == "risk":
                        response = "📊 **Risk Assessment Analysis**\n\n"
                        response += f"Analyzing risk metrics for {', '.join(state.stocks)}:"
                        st.markdown(response)
                        create_risk_charts(state)
                        
                    elif state.analysis_type == "swot":
                        response = "📋 **SWOT Analysis**\n\n"
                        st.markdown(response)
                        display_swot_analysis(state)
                        
                    elif state.analysis_type == "profit_calc":
                        response = "💰 **Profit Calculation Analysis**\n\n"
                        st.markdown(response)
                        display_profit_analysis(state)
                        
                    elif state.analysis_type == "backtest":
                        response = f"🚀 **Backtesting Analysis**\n\n"
                        response += f"Running backtest on {', '.join(state.backtest_stocks)} with {len(state.backtest_strategies)} strategies..."
                        response += f"\n\n**Stocks:** {', '.join(state.backtest_stocks)}"
                        response += f"\n**Strategies:** {', '.join(state.backtest_strategies)}"
                        response += f"\n**Amount per trade:** ₹{state.backtest_amount:,}"
                        response += f"\n**Duration:** {state.duration}"
                        st.markdown(response)
                        
                        with st.spinner("Running backtest... This may take a few minutes..."):
                            backtest_results = run_backtest(state)
                        
                        st.success(f"✅ Backtest completed! Generated {len(backtest_results['results'])} results.")
                        display_backtest_results(state)
                    
                    # Generate natural language summary
                    llm_response = generate_llm_response(prompt, state, st.session_state.messages)
                    st.info(llm_response)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response if 'response' in locals() else "Analysis complete.",
                        "analysis": {
                            "comparison_table": state.comparison_table,
                            "charts": lambda: visualize_combined_indicators(state)
                        }
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}\n\nPlease try rephrasing your query. Example: 'Compare TCS and INFY for 1 month'"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Collapsible sidebar
    with st.sidebar:
        # Sidebar toggle
        if st.button("📚", help="Toggle Examples", key="sidebar_toggle"):
            st.session_state.sidebar_expanded = not st.session_state.get('sidebar_expanded', True)
        
        if st.session_state.get('sidebar_expanded', True):
            st.header("📚 Query Examples")
            st.markdown("""
            **Comparisons:**
            - Compare TCS and INFY using RSI and MACD
            - Which is better, HDFC or ICICI?
            
            **Risk Analysis:**
            - Risk assessment of RELIANCE
            - Analyze volatility of tech stocks
            
            **SWOT Analysis:**
            - SWOT analysis of TCS
            - Strengths and weaknesses of AAPL
            
            **Profit Calculations:**
            - Capital needed for $5000 profit from MSFT
            - How much to invest for 10000 rupees profit?
            
            **Backtesting:**
            - Run backtest on TCS, INFY, HDFC with RSI, MACD, Bollinger
            - Test strategies on RELIANCE and WIPRO with 50000 amount
            - Backtest ICICI and SBIN with all strategies
            
            **Voice Input:**
            - Click 🎤 button next to text input
            - Speak naturally: "Analyze TCS"
            - Try: "Compare HDFC and ICICI" 
            - Say: "Risk analysis of RELIANCE"
            - Voice: "Backtest TCS with RSI strategy"
            
            **Technical Analysis:**
            - Show Bollinger Bands for GOOGL
            - Analyze AMZN with all indicators for 2 weeks
            """)
            
            st.divider()
            
            if st.button("🔄 Clear Chat"):
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": "👋 Chat cleared! How can I help you analyze stocks today?"
                }]
                st.session_state.analysis_state = None
                st.rerun()

if __name__ == "__main__":
    main()