import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
from tabulate import tabulate

class StockConfidenceAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = None
        
    def fetch_data(self, period="1y"):
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(period=period)
        
    def calculate_intervals(self, data):
        mean = data.mean()
        std = data.std()
        n = len(data)
        
        t_95 = stats.t.ppf(0.975, df=n-1)
        t_99 = stats.t.ppf(0.995, df=n-1)
        
        se = std / np.sqrt(n)
        
        ci_95_lower = mean - t_95 * se
        ci_95_upper = mean + t_95 * se
        ci_99_lower = mean - t_99 * se
        ci_99_upper = mean + t_99 * se
        
        return {
            'mean': mean,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'ci_99_lower': ci_99_lower,
            'ci_99_upper': ci_99_upper
        }
    
    def get_daily_analysis(self):
        return self.calculate_intervals(self.data['Close'])
    
    def get_weekly_analysis(self):
        weekly_data = self.data['Close'].resample('W').last()
        return self.calculate_intervals(weekly_data)
    
    def get_monthly_analysis(self):
        monthly_data = self.data['Close'].resample('M').last()
        return self.calculate_intervals(monthly_data)
    
    def get_complete_analysis(self):
        daily = self.get_daily_analysis()
        weekly = self.get_weekly_analysis()
        monthly = self.get_monthly_analysis()
        
        current_price = self.data['Close'].iloc[-1]
        
        return {
            'symbol': self.symbol.upper(),
            'current_price': current_price,
            'daily': daily,
            'weekly': weekly,
            'monthly': monthly
        }

def analyze_multiple_stocks(symbols):
    results = []
    
    for symbol in symbols:
        try:
            analyzer = StockConfidenceAnalyzer(symbol)
            analyzer.fetch_data()
            analysis = analyzer.get_complete_analysis()
            
            # 각 주기별 데이터를 플랫하게 만들기
            flat_data = {
                'Symbol': analysis['symbol'],
                'Current Price': f"{analysis['current_price']:.2f}",
                'Daily Mean': f"{analysis['daily']['mean']:.2f}",
                'Daily 95% CI': f"[{analysis['daily']['ci_95_lower']:.2f}, {analysis['daily']['ci_95_upper']:.2f}]",
                'Daily 99% CI': f"[{analysis['daily']['ci_99_lower']:.2f}, {analysis['daily']['ci_99_upper']:.2f}]",
                'Weekly Mean': f"{analysis['weekly']['mean']:.2f}",
                'Weekly 95% CI': f"[{analysis['weekly']['ci_95_lower']:.2f}, {analysis['weekly']['ci_95_upper']:.2f}]",
                'Weekly 99% CI': f"[{analysis['weekly']['ci_99_lower']:.2f}, {analysis['weekly']['ci_99_upper']:.2f}]",
                'Monthly Mean': f"{analysis['monthly']['mean']:.2f}",
                'Monthly 95% CI': f"[{analysis['monthly']['ci_95_lower']:.2f}, {analysis['monthly']['ci_95_upper']:.2f}]",
                'Monthly 99% CI': f"[{analysis['monthly']['ci_99_lower']:.2f}, {analysis['monthly']['ci_99_upper']:.2f}]"
            }
            results.append(flat_data)
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            
    return pd.DataFrame(results)

# 분석할 주식 리스트
stocks_to_analyze = ['SOXL', 'TQQQ', 'UPRO']

# 분석 실행
results_df = analyze_multiple_stocks(stocks_to_analyze)

# 결과 출력
print("\n레버리지 ETF 신뢰구간 분석")
print("=" * 100)
print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))

if __name__ == "__main__":
    # 분석할 주식 리스트 정의
    stocks_to_analyze = ['SOXL', 'TQQQ', 'UPRO']
    
    # 분석 실행 및 결과 출력
    results_df = analyze_multiple_stocks(stocks_to_analyze)
    print("\n레버리지 ETF 신뢰구간 분석")
    print("=" * 100)
    print(tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
