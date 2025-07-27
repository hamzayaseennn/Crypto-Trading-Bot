# Crypto Trading Bot 🤖📈

An intelligent cryptocurrency trading bot that combines machine learning predictions with technical analysis for automated trading on Binance. Features a modern GUI interface and comprehensive risk management.

## 🌟 Features

- **Machine Learning Integration**: Uses Linear Regression model for price movement prediction
- **Technical Analysis**: Implements SMA (Simple Moving Average) crossover strategies
- **Risk Management**: Built-in stop-loss and take-profit mechanisms
- **GUI Interface**: Modern Tkinter-based interface with Binance-themed design
- **Test Mode**: Safe testing environment using Binance testnet
- **Real-time Monitoring**: Live price updates and trading statistics
- **Manual Trading**: Execute manual buy/sell orders with custom parameters

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Binance account with API access
- Basic understanding of cryptocurrency trading

### Installation

1. **Clone the repository**
   ```bash
   git clone (https://github.com/hamzayaseennn/Crypto-Trading-Bot)
   cd crypto-trading-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   BINANCE_API_KEY=your_binance_api_key_here
   BINANCE_API_SECRET=your_binance_secret_key_here
   TRADING_PAIR=BTCUSDT
   QUANTITY=0.001
   STOP_LOSS_PERCENTAGE=2
   TAKE_PROFIT_PERCENTAGE=3
   ```

4. **Run the bot**
   
   **Command Line Interface:**
   ```bash
   python tradingbot.py
   ```
   
   **GUI Interface:**
   ```bash
   python bot_interface.py
   ```

## 📊 Trading Strategy

The bot employs a hybrid approach combining:

### Machine Learning Component
- **Model**: Linear Regression trained on historical price data
- **Features**: Returns, volatility, volume moving average, price moving average
- **Prediction**: Binary classification (UP/DOWN) with confidence scoring

### Technical Analysis
- **SMA Crossover**: 20-period and 50-period Simple Moving Average crossover
- **Signal Generation**: Combines ML predictions with technical indicators
- **Entry Conditions**: Requires both ML and technical confirmation

### Risk Management
- **Stop Loss**: Configurable percentage-based stop loss
- **Take Profit**: Automatic profit-taking at target levels
- **Position Sizing**: Fixed quantity trading with customizable amounts

## 🖥️ GUI Interface

The bot includes a sophisticated GUI with:

- **Real-time Price Display**: Live cryptocurrency prices
- **ML Predictions**: Current model predictions with confidence levels
- **Trading Statistics**: Win rate, total trades, profit/loss tracking
- **Manual Controls**: Execute manual trades with custom parameters
- **Live Logging**: Real-time trading activity logs

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BINANCE_API_KEY` | Your Binance API key | Required |
| `BINANCE_API_SECRET` | Your Binance secret key | Required |
| `TRADING_PAIR` | Cryptocurrency pair to trade | BTCUSDT |
| `QUANTITY` | Trade quantity | 0.001 |
| `STOP_LOSS_PERCENTAGE` | Stop loss percentage | 2 |
| `TAKE_PROFIT_PERCENTAGE` | Take profit percentage | 3 |

### Trading Parameters

- **Prediction Threshold**: 0.5 (50% confidence minimum)
- **Update Interval**: 5 minutes for ML model, 1 minute for price monitoring
- **Technical Indicators**: SMA20, SMA50 crossover strategy

## 🔒 Security Features

- **Testnet Mode**: Default operation on Binance testnet for safe testing
- **Environment Variables**: Sensitive data stored in `.env` files
- **Git Protection**: Comprehensive `.gitignore` prevents credential exposure
- **Error Handling**: Robust exception handling for API failures

## 📈 Performance Tracking

The bot tracks comprehensive statistics:

- Total number of trades executed
- Win/loss ratio and win rate percentage
- Total profit/loss across all trades
- Current position profit/loss
- Real-time performance metrics

## 🛡️ Risk Disclaimer

**⚠️ Important Warning:**

- This bot is for educational and testing purposes
- Cryptocurrency trading involves significant financial risk
- Past performance does not guarantee future results
- Always test thoroughly on testnet before live trading
- Never invest more than you can afford to lose
- The authors are not responsible for any financial losses

## 🔧 Development

### Project Structure

```
crypto-trading-bot/
├── tradingbot.py          # Core trading bot logic
├── bot_interface.py       # GUI interface
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

### Key Dependencies

- `python-binance`: Binance API integration
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning models
- `tkinter`: GUI framework
- `python-dotenv`: Environment variable management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Binance for providing comprehensive API documentation
- The Python trading community for inspiration and best practices
- Contributors and testers who help improve the bot


**Happy Trading! 🚀**

*Remember: Always trade responsibly and never risk more than you can afford to lose.*
