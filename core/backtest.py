import pandas as pd
import numpy as np
from feature_engine import FeatureEngine
from config import Config

class Backtester:
    @staticmethod
    def run_backtest(df: pd.DataFrame, initial_balance: float = 1000.0, payout_rate: float = 0.85):
        """
        Vectorized & Event-Driven Backtesting Engine.
        Simulates Fixed Time / Binary Trades with Risk Parameters.
        """
        df = FeatureEngine.generate_features(df)
        feature_cols = FeatureEngine.get_feature_columns()
        
        balance = initial_balance
        trades = []
        
        # Simple Backtest Loop
        for i in range(30, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i+1]
            
            # Basic Signal Condition
            signal = None
            if row['bullish_ob'] or row['liquidity_sweep_low'] or row['bos'] == 1:
                signal = "CALL"
            elif row['bearish_ob'] or row['liquidity_sweep_high'] or row['bos'] == -1:
                signal = "PUT"
                
            if signal:
                trade_amount = balance * 0.02 # 2% Risk per trade
                is_win = False
                
                if signal == "CALL" and next_row['Close'] > row['Close']:
                    is_win = True
                elif signal == "PUT" and next_row['Close'] < row['Close']:
                    is_win = True
                
                if is_win:
                    profit = trade_amount * payout_rate
                    balance += profit
                    trades.append({"type": signal, "result": "WIN", "profit": profit, "balance": balance})
                else:
                    balance -= trade_amount
                    trades.append({"type": signal, "result": "LOSS", "profit": -trade_amount, "balance": balance})

        total_trades = len(trades)
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        win_rate = round((wins / total_trades * 100), 2) if total_trades > 0 else 0.0
        net_profit = round(balance - initial_balance, 2)

        return {
            "initial_balance": initial_balance,
            "final_balance": round(balance, 2),
            "net_profit": net_profit,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "wins": wins,
            "losses": total_trades - wins
        }
