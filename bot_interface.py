import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from tradingbot import CryptoTrader
from datetime import datetime

class TradingBotInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Crypto Trading Bot Interface")
        self.root.geometry("1200x800")
        
        # Binance theme colors
        self.bg_color = "#1E2329"  # Dark background
        self.secondary_bg = "#2B3139"  # Slightly lighter background
        self.accent_color = "#F0B90B"  # Binance yellow
        self.text_color = "#EAECEF"  # Light text
        self.success_color = "#03A66D"  # Green for profits
        self.danger_color = "#CF304A"  # Red for losses
        
        # Configure root window
        self.root.configure(bg=self.bg_color)
        
        # Configure styles
        self.configure_styles()
        
        # Initialize trading bot
        self.trader = CryptoTrader()
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10", style='TFrame')
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Header
        self.create_header()
        
        # Trading Pair Info
        self.create_trading_pair_section()
        
        # Price and Prediction Section
        self.create_price_prediction_section()
        
        # Manual Trading Section
        self.create_manual_trading_section()
        
        # Trading Statistics Section
        self.create_statistics_section()
        
        # Log Section
        self.create_log_section()
        
        # Control Buttons
        self.create_control_buttons()
        
        # Initialize update thread
        self.running = True
        self.update_thread = threading.Thread(target=self.update_data)
        self.update_thread.daemon = True
        self.update_thread.start()

    def configure_styles(self):
        """Configure ttk styles for Binance theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background=self.bg_color)
        style.configure('TLabelframe', background=self.bg_color, foreground=self.text_color)
        style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.text_color)
        style.configure('TLabel', background=self.bg_color, foreground=self.text_color)
        style.configure('TButton', 
                       background=self.secondary_bg,
                       foreground=self.text_color,
                       padding=10)
        style.map('TButton',
                 background=[('active', self.accent_color)],
                 foreground=[('active', self.bg_color)])
        
        # Configure entry style
        style.configure('TEntry',
                       fieldbackground=self.secondary_bg,
                       foreground=self.text_color,
                       insertcolor=self.text_color)

    def create_header(self):
        """Create header section"""
        header_frame = ttk.Frame(self.main_frame, style='TFrame')
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Title
        title_label = ttk.Label(header_frame,
                              text="Crypto Trading Bot",
                              font=("Arial", 24, "bold"),
                              foreground=self.accent_color)
        title_label.pack(side="left")
        
        # Test Mode Warning
        warning_label = ttk.Label(header_frame,
                                text="⚠️ TEST MODE",
                                foreground=self.accent_color,
                                font=("Arial", 12, "bold"))
        warning_label.pack(side="right")

    def create_trading_pair_section(self):
        """Create trading pair information section"""
        pair_frame = ttk.LabelFrame(self.main_frame, text="Trading Pair Information", padding="10")
        pair_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Trading Pair
        ttk.Label(pair_frame, text="Trading Pair:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.pair_label = ttk.Label(pair_frame, text=self.trader.trading_pair, foreground=self.accent_color)
        self.pair_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Quantity
        ttk.Label(pair_frame, text="Trading Quantity:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.quantity_label = ttk.Label(pair_frame, text=f"{self.trader.quantity}", foreground=self.accent_color)
        self.quantity_label.grid(row=0, column=3, sticky=tk.W, padx=5)

    def create_price_prediction_section(self):
        """Create price and prediction section"""
        pred_frame = ttk.LabelFrame(self.main_frame, text="Price and Predictions", padding="10")
        pred_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Current Price
        ttk.Label(pred_frame, text="Current Price:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.price_label = ttk.Label(pred_frame, text="0.00", foreground=self.accent_color)
        self.price_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # ML Prediction
        ttk.Label(pred_frame, text="ML Prediction:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.ml_pred_label = ttk.Label(pred_frame, text="Waiting...", foreground=self.text_color)
        self.ml_pred_label.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Technical Analysis
        ttk.Label(pred_frame, text="Technical Analysis:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.tech_analysis_label = ttk.Label(pred_frame, text="Waiting...", foreground=self.text_color)
        self.tech_analysis_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Trading Signal
        ttk.Label(pred_frame, text="Trading Signal:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.signal_label = ttk.Label(pred_frame, text="Waiting...", foreground=self.text_color)
        self.signal_label.grid(row=1, column=3, sticky=tk.W, padx=5)

    def create_manual_trading_section(self):
        """Create manual trading controls section"""
        manual_frame = ttk.LabelFrame(self.main_frame, text="Manual Trading", padding="10")
        manual_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Quantity Entry
        ttk.Label(manual_frame, text="Quantity:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.quantity_var = tk.StringVar(value=str(self.trader.quantity))
        self.quantity_entry = ttk.Entry(manual_frame, textvariable=self.quantity_var, width=10, style='TEntry')
        self.quantity_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Stop Loss Entry
        ttk.Label(manual_frame, text="Stop Loss %:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.stop_loss_var = tk.StringVar(value=str(self.trader.stop_loss_percentage))
        self.stop_loss_entry = ttk.Entry(manual_frame, textvariable=self.stop_loss_var, width=10, style='TEntry')
        self.stop_loss_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Take Profit Entry
        ttk.Label(manual_frame, text="Take Profit %:").grid(row=0, column=4, sticky=tk.W, padx=5)
        self.take_profit_var = tk.StringVar(value=str(self.trader.take_profit_percentage))
        self.take_profit_entry = ttk.Entry(manual_frame, textvariable=self.take_profit_var, width=10, style='TEntry')
        self.take_profit_entry.grid(row=0, column=5, sticky=tk.W, padx=5)
        
        # Trading Buttons
        button_frame = ttk.Frame(manual_frame)
        button_frame.grid(row=1, column=0, columnspan=6, pady=10)
        
        self.buy_button = ttk.Button(button_frame, text="Buy", command=self.manual_buy, style='TButton')
        self.buy_button.grid(row=0, column=0, padx=5)
        
        self.sell_button = ttk.Button(button_frame, text="Sell", command=self.manual_sell, style='TButton')
        self.sell_button.grid(row=0, column=1, padx=5)
        
        self.close_button = ttk.Button(button_frame, text="Close Position", command=self.close_position, style='TButton')
        self.close_button.grid(row=0, column=2, padx=5)
        
        # Position Info
        self.position_label = ttk.Label(manual_frame, text="No Position", foreground=self.text_color)
        self.position_label.grid(row=2, column=0, columnspan=6, pady=5)
        
        # After creating buttons, set their initial state
        self.update_manual_buttons()

    def create_statistics_section(self):
        """Create trading statistics section"""
        stats_frame = ttk.LabelFrame(self.main_frame, text="Trading Statistics", padding="10")
        stats_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Total Trades
        ttk.Label(stats_frame, text="Total Trades:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.total_trades_label = ttk.Label(stats_frame, text="0", foreground=self.text_color)
        self.total_trades_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Win Rate
        ttk.Label(stats_frame, text="Win Rate:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.win_rate_label = ttk.Label(stats_frame, text="0%", foreground=self.text_color)
        self.win_rate_label.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Total P/L
        ttk.Label(stats_frame, text="Total P/L:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.total_pl_label = ttk.Label(stats_frame, text="0.00%", foreground=self.text_color)
        self.total_pl_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Current P/L
        ttk.Label(stats_frame, text="Current P/L:").grid(row=1, column=2, sticky=tk.W, padx=5)
        self.current_pl_label = ttk.Label(stats_frame, text="0.00%", foreground=self.text_color)
        self.current_pl_label.grid(row=1, column=3, sticky=tk.W, padx=5)

    def create_log_section(self):
        """Create log section"""
        log_frame = ttk.LabelFrame(self.main_frame, text="Trading Log", padding="10")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            width=100,
            bg=self.secondary_bg,
            fg=self.text_color,
            insertbackground=self.text_color,
            font=("Consolas", 10)
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.config(state=tk.DISABLED)

    def create_control_buttons(self):
        """Create control buttons"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Bot", command=self.start_bot, style='TButton')
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Bot", command=self.stop_bot, state=tk.DISABLED, style='TButton')
        self.stop_button.grid(row=0, column=1, padx=5)

    def manual_buy(self):
        """Execute manual buy order"""
        try:
            quantity = float(self.quantity_var.get())
            stop_loss = float(self.stop_loss_var.get())
            take_profit = float(self.take_profit_var.get())
            
            if self.trader.in_position:
                messagebox.showerror("Error", "Already in a position. Please close the current position before opening a new one.")
                return
            
            if self.trader.manual_buy(quantity, stop_loss, take_profit):
                self.add_log_entry("Manual buy order executed")
                self.update_position_info()
                self.update_manual_buttons()
            else:
                messagebox.showerror("Error", "Failed to execute buy order")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def manual_sell(self):
        """Execute manual sell order"""
        try:
            quantity = float(self.quantity_var.get())
            stop_loss = float(self.stop_loss_var.get())
            take_profit = float(self.take_profit_var.get())
            
            if self.trader.in_position:
                messagebox.showerror("Error", "Already in a position. Please close the current position before opening a new one.")
                return
            
            if self.trader.manual_sell(quantity, stop_loss, take_profit):
                self.add_log_entry("Manual sell order executed")
                self.update_position_info()
                self.update_manual_buttons()
            else:
                messagebox.showerror("Error", "Failed to execute sell order")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def close_position(self):
        """Close current position"""
        if self.trader.close_position():
            self.add_log_entry("Position closed")
            self.update_position_info()
            self.update_manual_buttons()
        else:
            messagebox.showerror("Error", "Failed to close position")

    def update_position_info(self):
        """Update position information display"""
        if self.trader.in_position:
            position_text = f"Position: {self.trader.position_type} | Entry: {self.trader.entry_price:.2f} | SL: {self.trader.stop_loss_price:.2f} | TP: {self.trader.take_profit_price:.2f}"
            color = self.success_color if self.trader.position_type == 'LONG' else self.danger_color
        else:
            position_text = "No Position"
            color = self.text_color
        self.position_label.config(text=position_text, foreground=color)

    def update_pl_colors(self, pl_value, label):
        """Update profit/loss label color based on value"""
        if pl_value > 0:
            label.config(foreground=self.success_color)
        elif pl_value < 0:
            label.config(foreground=self.danger_color)
        else:
            label.config(foreground=self.text_color)

    def update_manual_buttons(self):
        """Enable/disable manual trading buttons based on position state"""
        if self.trader.in_position:
            self.buy_button.config(state=tk.DISABLED)
            self.sell_button.config(state=tk.DISABLED)
            self.close_button.config(state=tk.NORMAL)
        else:
            self.buy_button.config(state=tk.NORMAL)
            self.sell_button.config(state=tk.NORMAL)
            self.close_button.config(state=tk.DISABLED)

    def update_data(self):
        """Update interface data periodically"""
        while self.running:
            try:
                # Update price
                current_price = self.trader.get_current_price()
                self.price_label.config(text=f"{current_price:.2f}")
                
                # Update ML prediction
                prediction = self.trader.predict_price_movement()
                if prediction:
                    pred_text = f"{prediction['prediction']} (Confidence: {prediction['confidence']:.2%})"
                    self.ml_pred_label.config(
                        text=pred_text,
                        foreground=self.success_color if prediction['prediction'] == 'UP' else self.danger_color
                    )
                
                # Update technical analysis
                df = self.trader.calculate_indicators()
                if df is not None:
                    tech_signal = "BUY" if df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] else "SELL"
                    self.tech_analysis_label.config(
                        text=tech_signal,
                        foreground=self.success_color if tech_signal == "BUY" else self.danger_color
                    )
                
                # Update trading signal
                if prediction and df is not None:
                    ml_signal = prediction['prediction'] == 'UP' and prediction['confidence'] > self.trader.prediction_threshold
                    tech_signal = df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1]
                    if ml_signal and tech_signal:
                        signal = "STRONG BUY"
                        color = self.success_color
                    elif ml_signal or tech_signal:
                        signal = "WEAK BUY"
                        color = self.accent_color
                    else:
                        signal = "SELL"
                        color = self.danger_color
                    self.signal_label.config(text=signal, foreground=color)
                
                # Update statistics
                stats = self.trader.get_trading_stats()
                self.total_trades_label.config(text=str(stats['total_trades']))
                self.win_rate_label.config(text=f"{stats['win_rate']:.2f}%")
                
                # Update P/L with colors
                self.update_pl_colors(stats['total_profit_loss'], self.total_pl_label)
                self.update_pl_colors(stats['current_profit_loss'], self.current_pl_label)
                self.total_pl_label.config(text=f"{stats['total_profit_loss']:.2f}%")
                self.current_pl_label.config(text=f"{stats['current_profit_loss']:.2f}%")
                
                # Update position info
                self.update_position_info()
                self.update_manual_buttons()
                
                # Add log entry
                self.add_log_entry(f"Price: {current_price:.2f} | ML: {prediction['prediction'] if prediction else 'N/A'} | Signal: {self.signal_label.cget('text')}")
                
            except Exception as e:
                self.add_log_entry(f"Error updating data: {str(e)}")
            
            time.sleep(5)  # Update every 5 seconds

    def add_log_entry(self, message):
        """Add entry to log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_bot(self):
        """Start the trading bot"""
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.add_log_entry("Bot started")
        
        # Start trading in a separate thread
        self.trading_thread = threading.Thread(target=self.trader.run)
        self.trading_thread.daemon = True
        self.trading_thread.start()

    def stop_bot(self):
        """Stop the trading bot"""
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.add_log_entry("Bot stopped")

    def run(self):
        """Run the interface"""
        self.root.mainloop()

if __name__ == "__main__":
    interface = TradingBotInterface()
    interface.run() 