import os
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load environment variables
load_dotenv()

"""
Crypto Trading Bot
Author: Hamza Yaseen
"""

class CryptoTrader:
    def __init__(self):
        # Initialize Binance testnet client
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret, testnet=True)  # Enable testnet
        
        # Trading parameters
        self.trading_pair = os.getenv('TRADING_PAIR', 'BTCUSDT')
        self.quantity = float(os.getenv('QUANTITY', '0.001'))
        self.stop_loss_percentage = float(os.getenv('STOP_LOSS_PERCENTAGE', '2'))
        self.take_profit_percentage = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '3'))
        
        # ML Model parameters
        self.model = None
        self.scaler = StandardScaler()
        self.prediction_threshold = 0.5  # Threshold for prediction confidence
        
        # Trading state
        self.in_position = False
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0
        self.initial_balance = 0
        self.final_balance = 0
        self.profit_loss = 0
        self.position_type = None  # 'LONG' or 'SHORT'
        
        # Trading statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit_loss = 0
        
        # Test mode flag
        self.test_mode = True
        print("Running in TEST MODE - No real trades will be executed!")
        
        # Initialize ML model
        self.initialize_ml_model()

    def initialize_ml_model(self):
        """Initialize and train the ML model"""
        try:
            # Get historical data for training
            klines = self.client.get_klines(
                symbol=self.trading_pair,
                interval=Client.KLINE_INTERVAL_5MINUTE,  # Changed to 5 minutes
                limit=500  # Get more data for better training
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Create features
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=24).std()
            df['volume_ma'] = df['volume'].rolling(window=24).mean()
            df['price_ma'] = df['close'].rolling(window=24).mean()
            
            # Create target variable (1 if price goes up in next period, 0 if down)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Drop NaN values
            df = df.dropna()
            
            # Prepare features for training
            features = ['returns', 'volatility', 'volume_ma', 'price_ma']
            X = df[features]
            y = df['target']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)
            
            print("ML model initialized and trained successfully!")
            
        except Exception as e:
            print(f"Error initializing ML model: {e}")
            self.model = None

    def predict_price_movement(self):
        """Predict whether price will go up or down"""
        try:
            if self.model is None:
                return None
                
            # Get recent data
            klines = self.client.get_klines(
                symbol=self.trading_pair,
                interval=Client.KLINE_INTERVAL_5MINUTE,  # Changed to 5 minutes
                limit=25  # Need 25 periods for features
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Create features
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=24).std()
            df['volume_ma'] = df['volume'].rolling(window=24).mean()
            df['price_ma'] = df['close'].rolling(window=24).mean()
            
            # Get latest features
            features = ['returns', 'volatility', 'volume_ma', 'price_ma']
            X = df[features].iloc[-1:].values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            confidence = abs(prediction - 0.5) * 2  # Convert to confidence score
            
            return {
                'prediction': 'UP' if prediction > 0.5 else 'DOWN',
                'confidence': confidence,
                'prediction_value': prediction
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    def get_trading_stats(self):
        """Get trading statistics"""
        current_price = self.get_current_price()
        current_pl = self.calculate_profit_loss(current_price) if self.in_position else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'total_profit_loss': self.total_profit_loss,
            'current_profit_loss': current_pl
        }

    def get_account_balance(self):
        """Get account balance for the trading pair"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] in self.trading_pair:
                    return float(balance['free'])
        except Exception as e:
            print(f"Error getting account balance: {e}")
        return 0

    def get_current_price(self):
        """Get current price of the trading pair"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.trading_pair)
            return float(ticker['price'])
        except Exception as e:
            print(f"Error getting current price: {e}")
        return 0

    def calculate_indicators(self):
        """Calculate technical indicators for trading decisions"""
        try:
            # Get historical klines/candlestick data
            klines = self.client.get_klines(
                symbol=self.trading_pair,
                interval=Client.KLINE_INTERVAL_1HOUR,
                limit=100
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert string values to float
            df['close'] = df['close'].astype(float)
            
            # Calculate Simple Moving Averages
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            
            return df
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        return None

    def calculate_profit_loss(self, current_price):
        """Calculate current profit/loss percentage"""
        if not self.in_position:
            return 0
        
        profit_loss = ((current_price - self.entry_price) / self.entry_price) * 100
        return profit_loss

    def place_order(self, side, quantity):
        """Place a market order"""
        try:
            if self.test_mode:
                print(f"TEST MODE: Would place {side} order for {quantity} {self.trading_pair}")
                # Simulate order placement
                return {
                    'symbol': self.trading_pair,
                    'side': side,
                    'quantity': quantity,
                    'status': 'TEST'
                }
            else:
                order = self.client.create_order(
                    symbol=self.trading_pair,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                print(f"Order placed: {order}")
                return order
        except Exception as e:
            print(f"Error placing order: {e}")
        return None

    def manual_buy(self, quantity=None, stop_loss=None, take_profit=None):
        """Execute a manual buy order with optional TP/SL"""
        try:
            if self.in_position:
                print("Already in a position. Close current position first.")
                return False

            current_price = self.get_current_price()
            quantity = quantity or self.quantity
            stop_loss = stop_loss or self.stop_loss_percentage
            take_profit = take_profit or self.take_profit_percentage

            # Place buy order
            order = self.place_order(SIDE_BUY, quantity)
            if order:
                self.in_position = True
                self.position_type = 'LONG'
                self.entry_price = current_price
                self.stop_loss_price = current_price * (1 - stop_loss / 100)
                self.take_profit_price = current_price * (1 + take_profit / 100)
                print(f"Buy order executed at {current_price}")
                print(f"Stop Loss: {self.stop_loss_price:.2f}")
                print(f"Take Profit: {self.take_profit_price:.2f}")
                return True
            return False
        except Exception as e:
            print(f"Error in manual buy: {e}")
            return False

    def manual_sell(self, quantity=None, stop_loss=None, take_profit=None):
        """Execute a manual sell order with optional TP/SL"""
        try:
            if self.in_position:
                print("Already in a position. Close current position first.")
                return False

            current_price = self.get_current_price()
            quantity = quantity or self.quantity
            stop_loss = stop_loss or self.stop_loss_percentage
            take_profit = take_profit or self.take_profit_percentage

            # Place sell order
            order = self.place_order(SIDE_SELL, quantity)
            if order:
                self.in_position = True
                self.position_type = 'SHORT'
                self.entry_price = current_price
                self.stop_loss_price = current_price * (1 + stop_loss / 100)
                self.take_profit_price = current_price * (1 - take_profit / 100)
                print(f"Sell order executed at {current_price}")
                print(f"Stop Loss: {self.stop_loss_price:.2f}")
                print(f"Take Profit: {self.take_profit_price:.2f}")
                return True
            return False
        except Exception as e:
            print(f"Error in manual sell: {e}")
            return False

    def close_position(self):
        """Close current position"""
        try:
            if not self.in_position:
                print("No position to close")
                return False

            current_price = self.get_current_price()
            side = SIDE_SELL if self.position_type == 'LONG' else SIDE_BUY
            
            order = self.place_order(side, self.quantity)
            if order:
                profit_loss = self.calculate_profit_loss(current_price)
                self.in_position = False
                self.position_type = None
                self.final_balance = self.get_account_balance()
                self.profit_loss = profit_loss
                self.total_trades += 1
                
                if profit_loss > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                self.total_profit_loss += profit_loss
                print(f"Position closed at {current_price}")
                print(f"Profit/Loss: {profit_loss:.2f}%")
                return True
            return False
        except Exception as e:
            print(f"Error closing position: {e}")
            return False

    def check_stop_loss_take_profit(self):
        """Check if stop loss or take profit has been hit"""
        if not self.in_position:
            return False

        current_price = self.get_current_price()
        profit_loss = self.calculate_profit_loss(current_price)
        
        if self.position_type == 'LONG':
            if current_price <= self.stop_loss_price:
                print(f"Stop loss triggered! Loss: {profit_loss:.2f}%")
                self.close_position()
                return True
            elif current_price >= self.take_profit_price:
                print(f"Take profit triggered! Profit: {profit_loss:.2f}%")
                self.close_position()
                return True
        else:  # SHORT position
            if current_price >= self.stop_loss_price:
                print(f"Stop loss triggered! Loss: {profit_loss:.2f}%")
                self.close_position()
                return True
            elif current_price <= self.take_profit_price:
                print(f"Take profit triggered! Profit: {profit_loss:.2f}%")
                self.close_position()
                return True
        return False

    def trading_strategy(self):
        """Implement trading strategy"""
        df = self.calculate_indicators()
        if df is None:
            return False

        current_price = self.get_current_price()
        
        # Get ML prediction
        prediction = self.predict_price_movement()
        if prediction:
            print(f"ML Prediction: {prediction['prediction']} (Confidence: {prediction['confidence']:.2%})")
        
        # Combine ML prediction with technical indicators
        if not self.in_position:
            # Check both ML prediction and technical indicators
            ml_buy_signal = prediction and prediction['prediction'] == 'UP' and prediction['confidence'] > self.prediction_threshold
            technical_buy_signal = df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] and df['SMA20'].iloc[-2] <= df['SMA50'].iloc[-2]
            
            if ml_buy_signal and technical_buy_signal:
                # Strong buy signal from both ML and technical analysis
                print("Strong buy signal detected from both ML and technical analysis!")
                order = self.place_order(SIDE_BUY, self.quantity)
                if order:
                    self.in_position = True
                    self.entry_price = current_price
                    self.stop_loss_price = current_price * (1 - self.stop_loss_percentage / 100)
                    self.take_profit_price = current_price * (1 + self.take_profit_percentage / 100)
        
        return self.check_stop_loss_take_profit()

    def get_trade_result(self):
        """Get the final trade result"""
        return {
            'profit_loss': self.profit_loss,
            'initial_balance': self.initial_balance,
            'final_balance': self.final_balance,
            'is_profit': self.profit_loss > 0
        }

    def run(self):
        """Main trading loop"""
        print(f"Starting trading bot for {self.trading_pair} in TEST MODE")
        self.initial_balance = self.get_account_balance()
        print(f"Initial balance: {self.initial_balance}")
        
        while True:
            try:
                if self.trading_strategy():
                    # If stop loss or take profit was hit, break the loop
                    result = self.get_trade_result()
                    print("\n=== Trade Completed ===")
                    print(f"Initial Balance: {result['initial_balance']:.8f}")
                    print(f"Final Balance: {result['final_balance']:.8f}")
                    print(f"Profit/Loss: {result['profit_loss']:.2f}%")
                    print("=====================\n")
                    break
                time.sleep(60)  # Wait for 1 minute before next iteration
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    trader = CryptoTrader()
    trader.run()


# HMAC-SHA-256 Key registered
# Save these values right now. They won't be shown ever again!

# API Key: SB3y3VgHmLtzcAT32kAHFIFYjdV1LP9WcoMx25sgs0QQb5IK6lVk9ZBn40eGRNsX

# Secret Key: 3Y9Z47p4tPMxW6svmLUqhN8uycZCOIjcHyFbUQQ8HYQ9kXCCokKfTkn3DKMNNZVd