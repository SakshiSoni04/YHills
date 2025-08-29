# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from textblob import TextBlob
import re
from collections import Counter
import time

# Set page configuration
st.set_page_config(page_title="Live Sentiment Dashboard", layout="wide")
st.title("📊 Live Sentiment Dashboard")
st.write("Monitor live sentiment trends for a keyword or brand.")

# -------------------------------
# Text preprocessing function
# -------------------------------
def preprocess_text(text):
    # Remove mentions, hashtags, URLs, and special characters
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Remove hashtags
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and remove extra spaces
    return text

# -------------------------------
# Sentiment analyzer and classifier (Positive/Negative only)
# -------------------------------
def analyze_sentiment(text):
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Analyze sentiment
    polarity = TextBlob(cleaned_text).sentiment.polarity
    
    # Classify sentiment (Positive/Negative only)
    if polarity > 0:
        sentiment = "Positive"
        sentiment_emoji = "😊"
    else:
        sentiment = "Negative"
        sentiment_emoji = "😠"
        
    return polarity, sentiment, sentiment_emoji

# -------------------------------
# Simulated live tweet fetcher
# -------------------------------
def get_fake_tweets(keyword, n=5):
    # Sample tweets (only positive and negative)
    sample_texts = [
        f"I love my new {keyword}, it's absolutely amazing! Best purchase ever!",
        f"{keyword} is the worst product I've ever used. Complete waste of money.",
        f"{keyword} has completely transformed my daily routine for the better!",
        f"I'm so disappointed with {keyword}. It broke after just one week.",
        f"Highly recommend {keyword} to everyone! Excellent quality and service.",
        f"Avoid {keyword} at all costs. Terrible customer support experience.",
        f"Absolutely in love with my {keyword}! Worth every penny!",
        f"{keyword} stopped working after 2 days. Very frustrated!",
        f"{keyword} has exceeded all my expectations. Fantastic product!",
        f"Regret buying {keyword}. Poor build quality and performance.",
        f"{keyword} is a game-changer! Can't imagine life without it now.",
        f"Never buying {keyword} again. Complete garbage.",
        f"{keyword} customer service is outstanding! They went above and beyond.",
        f"{keyword} is a scam. Don't fall for their marketing.",
        f"Best decision I ever made was purchasing {keyword}!"
    ]
    return sample_texts[:n]

# -------------------------------
# Initialize session state
# -------------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time", "text", "sentiment_score", "sentiment", "sentiment_emoji"])
if "last_update" not in st.session_state:
    st.session_state.last_update = datetime.now()
if "running" not in st.session_state:
    st.session_state.running = False
if "keyword" not in st.session_state:
    st.session_state.keyword = "iPhone"

# -------------------------------
# Sidebar for controls
# -------------------------------
with st.sidebar:
    st.header("Controls")
    keyword = st.text_input("Enter a keyword/brand:", st.session_state.keyword)
    refresh_rate = st.slider("Refresh every X seconds:", 5, 60, 10)
    tweets_per_refresh = st.slider("Tweets per refresh:", 1, 10, 3)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Monitoring" if not st.session_state.running else "Pause Monitoring"):
            st.session_state.running = not st.session_state.running
            st.session_state.keyword = keyword
            st.rerun()
    with col2:
        if st.button("Clear Data"):
            st.session_state.data = pd.DataFrame(columns=["time", "text", "sentiment_score", "sentiment", "sentiment_emoji"])
            st.rerun()
    
    # Display current status
    if st.session_state.running:
        st.success("Monitoring is ACTIVE")
        next_update = st.session_state.last_update + timedelta(seconds=refresh_rate)
        st.write(f"Next update: {next_update.strftime('%H:%M:%S')}")
    else:
        st.info("Monitoring is PAUSED")

# -------------------------------
# Main dashboard
# -------------------------------
# Update data if monitoring is running
if st.session_state.running:
    time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
    
    if time_since_update >= refresh_rate:
        # Fetch "live" tweets
        tweets = get_fake_tweets(keyword, n=tweets_per_refresh)

        # Analyze and append results
        new_data = []
        for t in tweets:
            score, sentiment, emoji = analyze_sentiment(t)
            new_data.append({
                "time": datetime.now(), 
                "text": t, 
                "sentiment_score": score,
                "sentiment": sentiment,
                "sentiment_emoji": emoji
            })

        df_new = pd.DataFrame(new_data)
        st.session_state.data = pd.concat([st.session_state.data, df_new], ignore_index=True)
        st.session_state.last_update = datetime.now()
        
        # Force update of the display
        st.rerun()

# Display metrics if we have data
if not st.session_state.data.empty:
    positive = len(st.session_state.data[st.session_state.data["sentiment"] == "Positive"])
    negative = len(st.session_state.data[st.session_state.data["sentiment"] == "Negative"])
    total = len(st.session_state.data)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Posts", total)
    col2.metric("Positive", f"{positive} 😊", f"{positive/total*100:.1f}%")
    col3.metric("Negative", f"{negative} 😠", f"{negative/total*100:.1f}%")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Trend", "Distribution", "Live Feed", "Raw Data"])
    
    with tab1:
        # Time series chart with better formatting
        st.subheader("Sentiment Trend Over Time")
        chart_data = st.session_state.data[["time", "sentiment_score"]].copy()
        chart_data["time"] = pd.to_datetime(chart_data["time"])
        chart_data = chart_data.set_index("time")
        
        # Calculate rolling average for smoother trend line
        chart_data["rolling_avg"] = chart_data["sentiment_score"].rolling(window=3, min_periods=1).mean()
        
        # Create a more visually appealing chart
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(chart_data.index, chart_data["sentiment_score"], 'o-', alpha=0.5, label='Individual Scores')
        ax.plot(chart_data.index, chart_data["rolling_avg"], 'r-', linewidth=2, label='Trend (3-point avg)')
        
        # Add horizontal line at zero to separate positive and negative
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, label='Neutral Line')
        
        # Fill area above zero (positive) and below zero (negative)
        ax.fill_between(chart_data.index, chart_data["rolling_avg"], 0, 
                        where=chart_data["rolling_avg"] >= 0, color='green', alpha=0.2, label='Positive Area')
        ax.fill_between(chart_data.index, chart_data["rolling_avg"], 0, 
                        where=chart_data["rolling_avg"] < 0, color='red', alpha=0.2, label='Negative Area')
        
        ax.set_ylabel("Sentiment Score")
        ax.set_xlabel("Time")
        ax.set_title(f"Sentiment Trend for '{keyword}'")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        sentiment_counts = st.session_state.data["sentiment"].value_counts()
        colors = ['#4CAF50', '#FF5252']  # Green, Red
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title("Sentiment Distribution")
        
        # Bar chart
        ax2.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        ax2.set_title("Sentiment Count")
        ax2.set_ylabel("Number of Posts")
        
        # Add value labels on bars
        for i, v in enumerate(sentiment_counts.values):
            ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        # Live feed of tweets with sentiment
        st.subheader("Live Feed")
        
        # Display latest tweets with sentiment
        for _, row in st.session_state.data.tail(10).sort_values("time", ascending=False).iterrows():
            # Create a colored box based on sentiment
            if row["sentiment"] == "Positive":
                color = "green"
                bg_color = "#e8f5e9"  # Light green
            else:
                color = "red"
                bg_color = "#ffebee"  # Light red
            
            # Display tweet in a container with colored border
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding: 10px; margin: 10px 0; background-color: {bg_color};">
                <div style="display: flex; justify-content: space-between;">
                    <span><b>{row['sentiment_emoji']} {row['sentiment']}</b></span>
                    <span style="color: gray; font-size: 0.8em;">{row['time'].strftime('%H:%M:%S')}</span>
                </div>
                <p>{row['text']}</p>
                <div style="color: gray; font-size: 0.8em;">Score: {row['sentiment_score']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        # Raw data table
        st.subheader("Raw Data")
        display_df = st.session_state.data[["time", "text", "sentiment", "sentiment_score"]].copy()
        display_df["time"] = display_df["time"].dt.strftime("%H:%M:%S")
        
        # Apply color coding to the sentiment column
        def color_sentiment(val):
            color = 'green' if val == 'Positive' else 'red'
            return f'color: {color}; font-weight: bold'
        
        styled_df = display_df.sort_values("time", ascending=False).style.applymap(
            color_sentiment, subset=['sentiment']
        )
        
        st.dataframe(styled_df)
        
        # Add download button
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"sentiment_data_{keyword}.csv",
            mime="text/csv",
        )

else:
    st.info("Click 'Start Monitoring' to begin collecting data.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Sentiment Analysis Dashboard | Powered by TextBlob | Positive/Negative Classification Only</p>
</div>
""", unsafe_allow_html=True)