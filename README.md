Based on the requirements, the following classes, functions, and methods will be necessary:

Classes:
- StockPredictionBot: The main class that will handle the prediction and user interface functionalities.

Functions:
- get_company_code(company_name: str) -> str: A function that takes a company name as input and returns the corresponding company code.
- retrieve_historical_data(company_code: str) -> pd.DataFrame: A function that retrieves historical stock data from Yahoo Finance and returns it as a pandas DataFrame.
- preprocess_data(data: pd.DataFrame) -> pd.DataFrame: A function that preprocesses the historical stock data by cleaning and transforming it.
- train_model(data: pd.DataFrame) -> Any: A function that trains a machine learning model using the preprocessed historical data and returns the trained model.
- predict_stock_price(model: Any, data: pd.DataFrame) -> Tuple[str, float, str]: A function that takes a trained model and current stock data as input and returns a tuple containing the predicted market direction (up or down), future price, and date.
- get_news_sentiment(news_article: str) -> str: A function that takes a news article as input and returns its sentiment (positive, negative, or neutral).

Methods:
- __init__(self): The constructor method that initializes the StockPredictionBot object.
- run(self): The method that runs the StockPredictionBot and displays the user interface.

The code implementation for the above classes, functions, and methods are as follows:

**requirements.txt**
