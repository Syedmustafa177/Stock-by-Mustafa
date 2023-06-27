from typing import Tuple, Any
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import messagebox
import nltk
nltk.download('vader_lexicon')



class StockPredictionBot:
    def __init__(self):
        self.company_code = ""
        self.company_name = ""
        self.model = None
        self.data = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.root = tk.Tk()
        self.root.title("Stock Prediction Bot")
        self.root.geometry("400x200")
        self.label = tk.Label(self.root, text="Enter company name:")
        self.label.pack()
        self.entry = tk.Entry(self.root)
        self.entry.pack()
        self.button = tk.Button(self.root, text="Predict", command=self.predict)
        self.button.pack()

    def run(self):
        self.root.mainloop()

    def get_company_code(self, company_name: str) -> str:
        """
        A function that takes a company name as input and returns the corresponding company code.
        """
        company_name = company_name.lower()
        company_list = pd.read_csv("company_list.csv")
        company_list["Name"] = company_list["Name"].str.lower()
        company_list["Symbol"] = company_list["Symbol"].str.lower()
        if company_name in company_list["Name"].values:
            self.company_name = company_name.title()
            return company_list.loc[company_list["Name"] == company_name, "Symbol"].iloc[0]
        else:
            matches = company_list[company_list["Name"].str.contains(company_name)]
            if len(matches) > 0:
                self.company_name = matches.iloc[0]["Name"].title()
                return matches.iloc[0]["Symbol"]
            else:
                messagebox.showerror("Error", "Company not found.")
                return ""

    def retrieve_historical_data(self, company_code: str) -> pd.DataFrame:
        """
        A function that retrieves historical stock data from Yahoo Finance and returns it as a pandas DataFrame.
        """
        data = yf.download(company_code, period="max")
        return data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        A function that preprocesses the historical stock data by cleaning and transforming it.
        """
        data = data.dropna()
        data["Market_Direction"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
        data = data.drop(["Open", "High", "Low", "Adj Close", "Volume"], axis=1)
        return data

    def train_model(self, data: pd.DataFrame) -> Any:
        """
        A function that trains a machine learning model using the preprocessed historical data and returns the trained model.
        """
        X = data.drop(["Market_Direction"], axis=1)
        y = data["Market_Direction"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy < 0.9:
            messagebox.showwarning("Warning", "Prediction accuracy is below 90%.")
        return model

    def predict_stock_price(self, model: Any, data: pd.DataFrame) -> Tuple[str, float, str]:
        """
        A function that takes a trained model and current stock data as input and returns a tuple containing the predicted market direction (up or down), future price, and date.
        """
        current_data = data.iloc[-1, :-1].values.reshape(1, -1)
        prediction = model.predict(current_data)[0]
        future_price = data.iloc[-1, -2]
        future_date = data.index[-1].strftime("%Y-%m-%d")
        if prediction == 1:
            market_direction = "Up"
            future_price += (future_price * 0.01)
        else:
            market_direction = "Down"
            future_price -= (future_price * 0.01)
        return market_direction, future_price, future_date

    def get_news_sentiment(self, news_article: str) -> str:
        """
        A function that takes a news article as input and returns its sentiment (positive, negative, or neutral).
        """
        sentiment_scores = self.sentiment_analyzer.polarity_scores(news_article)
        if sentiment_scores["compound"] >= 0.05:
            return "Positive"
        elif sentiment_scores["compound"] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def predict(self):
        self.company_code = self.get_company_code(self.entry.get())
        if self.company_code != "":
            self.data = self.preprocess_data(self.retrieve_historical_data(self.company_code))
            self.model = self.train_model(self.data)
            market_direction, future_price, future_date = self.predict_stock_price(self.model, self.data)
            messagebox.showinfo("Prediction", f"Market direction: {market_direction}\nFuture price: {future_price}\nDate: {future_date}")


bot = StockPredictionBot()
bot.run()
