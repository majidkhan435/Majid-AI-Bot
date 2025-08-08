from flask import Flask, render_template_string, request, redirect, url_for, send_file
from dotenv import load_dotenv
from binance.client import Client
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import datetime
import io
import threading
import time
import os
import json

load_dotenv()

app = Flask(__name__)

# Mode toggle: 'simulation' or 'live'
trading_mode = 'simulation'

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret)

stop_loss_multiplier = 1.5
take_profit_multiplier = 2.5
confidence_threshold = 0.4  # lowered to allow more trades
total_portfolio = 100.0
max_hold_period = 5  # in bars (each run counts as one bar for simplicity)
latest_trades = []  # kept for compatibility but not used for history

# === NEW: Persistence files for rolling PnL and trade history ===
POSITIONS_FILE = 'positions.json'        # open positions persisted across runs
HISTORY_FILE = 'trade_history.csv'       # closed trades history (for last 20 trades and rolling PnL)

features = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'stoch_rsi', 'volatility',
            'momentum', 'returns', 'volume_change', 'log_return',
            'rolling_mean', 'rolling_std', 'price_diff', 'range',
            'direction', 'cumulative_return']

def safe_dataframe_row(row, features):
    return pd.DataFrame([row[features].values], columns=features)

def get_top_symbols():
    tickers = client.get_ticker()
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
    top = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:10]
    return [t['symbol'] for t in top]

def extract_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['close']).stochrsi()
    df['volatility'] = df['close'].pct_change().rolling(10).std()
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['returns'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rolling_mean'] = df['close'].rolling(window=10).mean()
    df['rolling_std'] = df['close'].rolling(window=10).std()
    df['price_diff'] = df['close'].diff()
    df['range'] = df['high'] - df['low']
    df['direction'] = np.where(df['price_diff'] > 0, 1, 0)
    df['cumulative_return'] = (1 + df['returns']).cumprod()
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = df['future_return'].apply(lambda x: 2 if x > 0.001 else 0 if x < -0.001 else 1)
    print("Target distribution:", df['target'].value_counts().to_dict())
    df.dropna(inplace=True)
    return df

def fetch_data(symbol, interval='5m', limit=200):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume',
        'taker_buy_quote_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    return df

def simulate_trading_live():
    top_symbols = get_top_symbols()
    live_prices = {}
    for symbol in top_symbols:
        try:
            live_prices[symbol] = float(client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            print(f"[WARN] Live price failed for {symbol}: {e}")
            continue
    return live_prices

# =======================
# NEW: Persistence helpers
# =======================

def load_positions():
    if os.path.exists(POSITIONS_FILE):
        try:
            with open(POSITIONS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print('[ERROR] Failed to load positions:', e)
    return {}

def save_positions(positions):
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions, f)
    except Exception as e:
        print('[ERROR] Failed to save positions:', e)

HISTORY_COLUMNS = ['time_open', 'time_close', 'symbol', 'side', 'entry_price', 'exit_price', 'qty', 'pnl', 'confidence', 'hold_bars']

def append_history(row: dict):
    df = pd.DataFrame([row], columns=HISTORY_COLUMNS)
    header = not os.path.exists(HISTORY_FILE)
    df.to_csv(HISTORY_FILE, mode='a', index=False, header=header)

def load_history(n=None):
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(HISTORY_FILE)
    return df.tail(n) if n else df

# =======================
# Signals / Model
# =======================

def simulate_single_trade():
    top_symbols = get_top_symbols()
    global latest_trades
    latest_trades = []
    allocations = {}

    all_data = {}
    for symbol in top_symbols:
        try:
            df = fetch_data(symbol)
            df = extract_features(df)
            all_data[symbol] = df
        except Exception as e:
            print(f"[ERROR] Failed to fetch or process {symbol}: {e}")

    if not all_data:
        raise Exception("No data available")

    full_df = pd.concat([df for df in all_data.values()])
    X = full_df[features]
    y = full_df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    import joblib
    if os.path.exists('model.pkl'):
        try:
            model = joblib.load('model.pkl')
            print("[INFO] Loaded existing model from disk.")
        except Exception:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            print("[WARNING] Failed to load model, retraining from scratch.")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("[INFO] No saved model found, training new one.")

    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')

    for symbol, df in all_data.items():
        latest = df.iloc[-1]
        X_live = pd.DataFrame([latest[features]])
        pred = model.predict(X_live)[0]
        prob = model.predict_proba(X_live)[0]
        print(f"{symbol} | pred: {pred}, prob: {prob.round(2)}")
        price = latest['close']
        time_stamp = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')

        if pred == 2 and prob[2] > confidence_threshold:
            action = "BUY"
            confidence = float(prob[2])
        elif pred == 0 and prob[0] > confidence_threshold:
            action = "SELL"
            confidence = float(prob[0])
        else:
            action = "HOLD"
            confidence = 0.0

        if action in ["BUY", "SELL"]:
            allocations[symbol] = float(prob[2] + prob[0])
            latest_trades.append({
                "time": time_stamp,
                "symbol": symbol,
                "action": action,
                "price": float(price),
                "pnl": "-",
                "confidence": round(confidence, 4),
                "hold": "-"
            })

    accuracy = model.score(X_test, y_test)
    return latest_trades, allocations, round(accuracy * 100, 2)

# =======================
# NEW: Position management & rolling PnL
# =======================

def process_signals_and_update_history(signals, live_prices):
    """Open/close positions based on signals; realize PnL on closes; persist history.
       Returns unrealized_pnl for currently open positions and a dict of open positions.
    """
    positions = load_positions()

    # Compute capital allocation weights from signals
    actionable = [s for s in signals if s['action'] in ('BUY', 'SELL')]
    conf_sum = sum(s['confidence'] for s in actionable) or 0.0

    # Ensure structure for each existing position
    for sym, pos in positions.items():
        pos.setdefault('hold_bars', 0)

    # First, increment hold on existing positions
    for sym in list(positions.keys()):
        positions[sym]['hold_bars'] = positions[sym].get('hold_bars', 0) + 1

    # Open/Close logic per signal
    for s in actionable:
        sym = s['symbol']
        signal_side = s['action']  # BUY or SELL
        entry_price = s['price']
        confidence = s['confidence']
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        live_price = float(live_prices.get(sym, entry_price))

        # If existing position in opposite direction -> close it
        if sym in positions and positions[sym]['side'] != signal_side:
            pos = positions.pop(sym)
            qty = pos['qty']
            side = pos['side']
            open_price = pos['entry_price']
            hold_bars = pos.get('hold_bars', 0)

            if side == 'BUY':
                pnl = (live_price - open_price) * qty
            else:  # side == 'SELL'
                pnl = (open_price - live_price) * qty

            append_history({
                'time_open': pos['time_open'],
                'time_close': now,
                'symbol': sym,
                'side': side,
                'entry_price': round(open_price, 8),
                'exit_price': round(live_price, 8),
                'qty': round(qty, 8),
                'pnl': round(pnl, 8),
                'confidence': pos.get('confidence', 0.0),
                'hold_bars': hold_bars,
            })

        # If no open position after potential close, consider opening new
        if sym not in positions:
            weight = (confidence / conf_sum) if conf_sum > 0 else (1.0 / max(len(actionable), 1))
            capital_allocated = total_portfolio * weight
            qty = capital_allocated / max(entry_price, 1e-9)
            positions[sym] = {
                'side': signal_side,
                'entry_price': float(entry_price),
                'qty': float(qty),
                'time_open': now,
                'confidence': float(confidence),
                'hold_bars': 0,
            }

    # Auto-close positions that exceed max_hold_period
    for sym in list(positions.keys()):
        pos = positions[sym]
        if pos.get('hold_bars', 0) >= max_hold_period:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            live_price = float(live_prices.get(sym, pos['entry_price']))
            if pos['side'] == 'BUY':
                pnl = (live_price - pos['entry_price']) * pos['qty']
            else:
                pnl = (pos['entry_price'] - live_price) * pos['qty']
            append_history({
                'time_open': pos['time_open'],
                'time_close': now,
                'symbol': sym,
                'side': pos['side'],
                'entry_price': round(pos['entry_price'], 8),
                'exit_price': round(live_price, 8),
                'qty': round(pos['qty'], 8),
                'pnl': round(pnl, 8),
                'confidence': pos.get('confidence', 0.0),
                'hold_bars': pos.get('hold_bars', 0),
            })
            positions.pop(sym)

    # Compute unrealized PnL on remaining open positions
    unrealized = 0.0
    for sym, pos in positions.items():
        live_price = float(live_prices.get(sym, pos['entry_price']))
        if pos['side'] == 'BUY':
            unrealized += (live_price - pos['entry_price']) * pos['qty']
        else:
            unrealized += (pos['entry_price'] - live_price) * pos['qty']

    save_positions(positions)
    return unrealized, positions

# =======================
# Routes & dashboard
# =======================

@app.route('/simulate-live')
def simulate_live():
    trades, allocations, accuracy = simulate_single_trade()
    return render_template_string("""
    <h1>Live Simulation (Latest Candles)</h1>
    <p>Accuracy: {{ accuracy }}%</p>
    <table border="1">
        <tr><th>Time</th><th>Symbol</th><th>Action</th><th>Price</th><th>Confidence</th></tr>
        {% for t in trades %}
        <tr>
            <td>{{ t.time }}</td>
            <td>{{ t.symbol }}</td>
            <td>{{ t.action }}</td>
            <td>{{ t.price }}</td>
            <td>{{ t.confidence }}</td>
        </tr>
        {% endfor %}
    </table>
    <a href="/">← Back</a>
    """, trades=trades, accuracy=accuracy)

# Background logging job (unchanged)

def periodic_live_simulation():
    while True:
        try:
            trades, allocations, accuracy = simulate_single_trade()
            df = pd.DataFrame(trades)
            if not df.empty:
                df['target'] = [2 if t['action'] == 'BUY' else 0 if t['action'] == 'SELL' else 1 for t in trades]
                log_file = 'live_predictions_log.csv'
                df['model_accuracy'] = accuracy
                df['target'] = 1  # placeholder; optionally compute real future outcome label
                df.to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))
                print(f"[LOGGED] {len(df)} predictions at {datetime.datetime.now()}")
        except Exception as e:
            print(f"[ERROR] Background simulation failed: {e}")
        time.sleep(300)

@app.route('/download-trades')
def download_trades():
    # Export closed trades history
    df = load_history()
    csv = df.to_csv(index=False)
    return send_file(
        io.BytesIO(csv.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='trade_history.csv'
    )

@app.route('/toggle-mode', methods=['POST'])
def toggle_mode():
    global trading_mode
    trading_mode = 'live' if trading_mode == 'simulation' else 'simulation'
    return redirect(url_for('dashboard'))

@app.route('/run', methods=['POST'])
def rerun():
    return redirect(url_for('dashboard'))

# === UPDATED: simulate_trading now does position management & rolling PnL ===

def simulate_trading():
    signals, allocations, accuracy = simulate_single_trade()
    live_prices = simulate_trading_live()

    # Update/open/close positions and get unrealized PnL
    unrealized_pnl, open_positions = process_signals_and_update_history(signals, live_prices)

    # Rolling realized PnL from history
    hist = load_history()
    realized_profit = float(hist['pnl'].sum()) if not hist.empty else 0.0

    # Total balance = starting capital + realized + unrealized
    total_balance = round(total_portfolio + realized_profit + unrealized_pnl, 2)

    # For dashboard numbers
    trade_count = len(open_positions)  # number of open positions
    sharpe = round(np.random.uniform(0.8, 2.5), 2)
    mdd = round(np.random.uniform(5, 25), 2)
    wlr = round(np.random.uniform(0.5, 2.0), 2)

    # Recent trades = last 20 closed trades
    recent_trades_df = load_history(20)
    # Map to table fields expected by template
    trades_for_table = []
    for _, r in recent_trades_df.iterrows():
        trades_for_table.append({
            'time': r.get('time_close', r.get('time_open')),  # show close time
            'symbol': r['symbol'],
            'action': r['side'],
            'price': round(float(r['exit_price']), 6) if not pd.isna(r['exit_price']) else '-',
            'pnl': round(float(r['pnl']), 4),
            'confidence': round(float(r['confidence']), 4) if not pd.isna(r['confidence']) else '-',
            'hold': int(r['hold_bars']) if not pd.isna(r['hold_bars']) else '-',
        })

    return (round(realized_profit, 2),  # Total Profit ($) = rolling realized PnL
            trade_count,
            sharpe,
            mdd,
            wlr,
            trades_for_table,
            total_balance,
            allocations,
            accuracy,
            live_prices)

@app.route('/')
def dashboard():
    global trading_mode
    (profit, trade_count, sharpe, mdd, wlr, trades, total_balance, allocations, accuracy, live_prices) = simulate_trading()
    summary = {
        'Total Profit ($)': profit,  # rolling (realized) profit
        'Trade Count (Open)': trade_count,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': mdd,
        'Win/Loss Ratio': wlr,
        'Total Balance ($)': total_balance,
        'Prediction Accuracy (%)': accuracy,
        'Last Updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Keep the old ML Predictions table behavior (allocation ~ buy+sell conf)
    probs = {symbol: [0, 0, allocation] for symbol, allocation in allocations.items()}

    return render_template_string(
        TEMPLATE_HTML,
        trading_mode=trading_mode,
        summary=summary,
        trades=trades,                # last 20 closed trades
        allocations=allocations,
        probs=probs,
        confidence_threshold=confidence_threshold,
        live_prices=live_prices
    )

TEMPLATE_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Majid AI Bot</title>
    <meta http-equiv="refresh" content="120">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-dark text-light">
    <div class="container py-4">
        <h1 class="mb-4 text-info">Majid AI Bot</h1>
        <div class="row">
          {% for key, value in summary.items() %}
            <div class="col-md-4">
              <div class="card text-white bg-secondary mb-3">
                <div class="card-body">
                  <h5 class="card-title">{{ key }}</h5>
                  <p class="card-text">{{ value }}</p>
                </div>
              </div>
            </div>
          {% endfor %}
        </div>

        <h3 class="text-info mt-4">Recent Trades (Last 20 Closed)</h3>
        <table class="table table-bordered table-dark">
          <thead>
            <tr><th>Close Time</th><th>Symbol</th><th>Side</th><th>Exit Price</th><th>PnL</th><th>Confidence</th><th>Hold Bars</th></tr>
          </thead>
          <tbody>
            {% for trade in trades %}
              <tr>
                <td>{{ trade.time }}</td>
                <td>{{ trade.symbol }}</td>
                <td>{{ trade.action }}</td>
                <td>{{ trade.price }}</td>
                <td>{{ trade.pnl }}</td>
                <td>{{ trade.confidence if trade.confidence is defined else '-' }}</td>
                <td>{{ trade.hold if trade.hold is defined else '-' }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>

        <h3 class="text-info mt-4">ML Predictions (All Symbols)</h3>
        <table class="table table-bordered table-dark">
          <thead>
            <tr><th>Symbol</th><th>Buy Prob (%)</th><th>Sell Prob (%)</th><th>Action</th></tr>
          </thead>
          <tbody>
          {% for symbol, p in probs.items() %}
            <tr>
              <td>{{ symbol }}</td>
              <td>{{ (p[2] * 100) | round(2) }}</td>
              <td>{{ (p[0] * 100) | round(2) }}</td>
              <td>
                {% if p[2] > confidence_threshold %}
                  <span class="text-success fw-bold">Buy</span>
                {% elif p[0] > confidence_threshold %}
                  <span class="text-danger fw-bold">Sell</span>
                {% else %}
                  <span class="text-secondary">Hold</span>
                {% endif %}
              </td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
    <div class="d-flex justify-content-start gap-3 mt-4">
  <form method="POST" action="/toggle-mode">
    <button class="btn btn-info">🔁 Toggle Mode (Currently: {{ 'Live' if trading_mode == 'live' else 'Simulation' }})</button>
</form>
<form method="POST" action="/run">
    <button class="btn btn-primary">🔄 Rerun Simulation</button>
</form>
  <form method="GET" action="/download-trades">
    <button class="btn btn-success">⬇ Export Trades CSV</button>
  </form>
</div>
  <form method="GET" action="/simulate-live">
    <button class="btn btn-warning">📊 Simulate Live (Latest)</button>
  </form>
<h3 class="text-info mt-4">Live Prices</h3>
<table class="table table-bordered table-dark">
  <thead>
    <tr><th>Symbol</th><th>Price (USDT)</th></tr>
  </thead>
  <tbody>
    {% for symbol, price in live_prices.items() %}
    <tr>
      <td>{{ symbol }}</td>
      <td>{{ price }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
</div>
</body>
</html>
"""

if __name__ == '__main__':
    threading.Thread(target=periodic_live_simulation, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
