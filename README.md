# Investment Portfolio & Crypto Prediction Web App

This is a Streamlit web application that allows you to manage your cryptocurrency portfolio and predict future prices of various coins.

## How to Run the Application

1.  **Ensure all dependencies are installed:**
    You have a `requirements.txt` file that lists all necessary libraries. Make sure they are installed in your Python environment. You can install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit application:**
    Open your terminal, navigate to the directory containing `main.py`, and run the following command:
    ```bash
    streamlit run main.py
    ```

    Your web browser should open with the application running.

## Features

### Price Prediction

-   Enter the ID of a cryptocurrency (e.g., `bitcoin`, `ethereum`).
-   Select the number of days you want to forecast.
    -   A 1-day forecast uses a simple Linear Regression model.
    -   Forecasts for 7 days or more use the more advanced Prophet AI model.
-   View the predicted price, model accuracy metrics, and a chart visualizing the forecast.

### Portfolio Management

-   Add coins and their quantities to your investment portfolio.
-   The application will fetch the current market prices from the CoinGecko API.
-   View the total value of your portfolio and a detailed breakdown of your holdings.
-   Remove coins from your portfolio.

**Note:** The portfolio data is stored in the session and will be reset if you close the browser tab or restart the application.