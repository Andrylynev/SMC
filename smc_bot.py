import asyncio
import ccxt
import pandas as pd
import numpy as np
from aiogram import Bot
import matplotlib.pyplot as plt
import io
from aiogram.types import BufferedInputFile
import mplfinance as mpf
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

# ======= Настройки =======
API_TOKEN = "7151895481:AAFIiyDtjPzzu7lRUClaXNRsiZ2jMOvrWQY"
TG_CHAT_ID = "1976131773"
TIMEFRAMES = ["1h", "30m", "15m", "5m"]
MIN_BARS = 100
HISTORY_LIMIT = 600

COLORS = dict(
    bull='#089981', bear='#F23645',
    swing_bull='#00ff68', swing_bear='#ff0008',
    internal_bull='#3179f5', internal_bear='#f77c80',
    bos='#2157f3', choch='#ff9900',
    eqh='#e100ff', eql='#00fff3',
    ob_bull='#1848cc', ob_bear='#b22833',
    fvg_bull='#00ff68', fvg_bear='#ff0008',
    daily='#2157f3', weekly='#F23645', monthly='#878b94',
    premium='#F23645', discount='#089981', eq='#878b94'
)

def find_pivots(series, left, right, mode='high'):
    pivots = [np.nan]*len(series)
    for i in range(left, len(series)-right):
        center = series.iloc[i]
        left_window = series.iloc[i-left:i]
        right_window = series.iloc[i+1:i+1+right]
        if mode == 'high' and all(center > left_window) and all(center > right_window):
            pivots[i] = center
        elif mode == 'low' and all(center < left_window) and all(center < right_window):
            pivots[i] = center
    return np.array(pivots)

def find_eq(series, threshold, atr, n=3, mode='high'):
    eq = [np.nan] * len(series)
    for i in range(n, len(series)):
        window = series.iloc[i-n+1:i+1]
        if mode == 'high':
            cond = (window.max() - window.min()) < threshold * atr.iloc[i]
        else:
            cond = (window.max() - window.min()) < threshold * atr.iloc[i]
        if cond:
            eq[i] = window.mean()
    return np.array(eq)

def premium_discount(df, left, right):
    highs = df['Pivot_High']
    lows = df['Pivot_Low']
    ph = np.nan
    pl = np.nan
    zone = []
    for i in range(len(df)):
        if not np.isnan(highs.iloc[i]):
            ph = highs.iloc[i]
        if not np.isnan(lows.iloc[i]):
            pl = lows.iloc[i]
        if not (np.isnan(ph) or np.isnan(pl)):
            zone.append((ph, pl))
        else:
            zone.append((np.nan, np.nan))
    z = pd.DataFrame(zone, columns=['swing_high', 'swing_low'], index=df.index)
    df['prem'] = z['swing_high']
    df['disc'] = z['swing_low']
    df['eq'] = (z['swing_high'] + z['swing_low']) / 2
    return df

def detect_bos_choch(df):
    df['BOS'] = None
    df['CHoCH'] = None
    trend = 0
    last_high = np.nan
    last_low = np.nan
    for i in range(1, len(df)):
        if not np.isnan(df['Pivot_High'].iloc[i]):
            last_high = df['Pivot_High'].iloc[i]
        if not np.isnan(df['Pivot_Low'].iloc[i]):
            last_low = df['Pivot_Low'].iloc[i]
        if pd.notna(last_high) and df['close'].iloc[i] > last_high:
            if trend == -1:
                df.loc[df.index[i], 'CHoCH'] = 'bullish'
            else:
                df.loc[df.index[i], 'BOS'] = 'bullish'
            trend = 1
        elif pd.notna(last_low) and df['close'].iloc[i] < last_low:
            if trend == 1:
                df.loc[df.index[i], 'CHoCH'] = 'bearish'
            else:
                df.loc[df.index[i], 'BOS'] = 'bearish'
            trend = -1
    return df

def order_blocks(df, mode='swing'):
    blocks = []
    for i in range(len(df)):
        if mode == 'swing':
            if not np.isnan(df['Pivot_High'].iloc[i]):
                price1 = df['Pivot_High'].iloc[i]
                price2 = df['close'].iloc[i]
                blocks.append({'idx': i, 'top': price1, 'bottom': price2, 'bull': False})
            if not np.isnan(df['Pivot_Low'].iloc[i]):
                price1 = df['Pivot_Low'].iloc[i]
                price2 = df['close'].iloc[i]
                blocks.append({'idx': i, 'top': price2, 'bottom': price1, 'bull': True})
    return blocks

def fair_value_gaps(df, min_size=0.5):
    fvgs = []
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            size = df['low'].iloc[i] - df['high'].iloc[i-2]
            if size > min_size * df['ATR'].iloc[i]:
                fvgs.append({'start': i-2, 'end': i, 'top': df['low'].iloc[i], 'bottom': df['high'].iloc[i-2], 'bull': True})
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            size = df['low'].iloc[i-2] - df['high'].iloc[i]
            if size > min_size * df['ATR'].iloc[i]:
                fvgs.append({'start': i-2, 'end': i, 'top': df['low'].iloc[i-2], 'bottom': df['high'].iloc[i], 'bull': False})
    return fvgs

def calc_atr(df, period=200):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def plot_smc_full(df, pair="BTCUSDT", tf="15m", signal_str=""):
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import io
    import matplotlib.pyplot as plt

    df = df.copy()
    df['ATR'] = calc_atr(df)
    if len(df) > 350:
        df = df.iloc[-350:]
    idx_list = df.index.to_list()
    df['Pivot_High'] = find_pivots(df['high'], left=5, right=5, mode='high')
    df['Pivot_Low']  = find_pivots(df['low'],  left=5, right=5, mode='low')
    df['Internal_High'] = find_pivots(df['high'], left=2, right=2, mode='high')
    df['Internal_Low']  = find_pivots(df['low'],  left=2, right=2, mode='low')
    df['EQH'] = find_eq(df['high'], threshold=0.1, atr=df['ATR'], n=3, mode='high')
    df['EQL'] = find_eq(df['low'],  threshold=0.1, atr=df['ATR'], n=3, mode='low')
    df = premium_discount(df, 5, 5)
    df = detect_bos_choch(df)
    ob_blocks = order_blocks(df, mode='swing')
    fvg_blocks = fair_value_gaps(df, min_size=0.5)
    colors = []
    trend = 0
    last_high = np.nan
    last_low = np.nan
    for i, row in df.iterrows():
        if not np.isnan(row['Pivot_High']):
            last_high = row['Pivot_High']
        if not np.isnan(row['Pivot_Low']):
            last_low = row['Pivot_Low']
        if pd.notna(last_high) and row['close'] > last_high:
            trend = 1
        elif pd.notna(last_low) and row['close'] < last_low:
            trend = -1
        colors.append(COLORS['bull'] if trend == 1 else COLORS['bear'])
    df['candle_color'] = colors
    df_mpf = df[['open', 'high', 'low', 'close']].copy()
    df_mpf['Volume'] = 1
    mc = mpf.make_marketcolors(
        up=COLORS['bull'], down=COLORS['bear'],
        edge=COLORS['bull'], wick=COLORS['bull'],
        volume='in', ohlc='i'
    )
    s = mpf.make_mpf_style(marketcolors=mc, gridcolor="#222", gridstyle="--", rc={'axes.labelsize': 8})
    fig, ax = mpf.plot(
        df_mpf, type='candle', style=s,
        figsize=(14, 7),
        title=f'SMC {pair} {tf} {signal_str}',
        returnfig=True, ylabel='Price',
        volume=False, warn_too_much_data=10000,
    )
    ax = fig.axes[0]
    last_trend = None
    for i, row in df.iterrows():
        xpos = mdates.date2num(idx_list[i])
        if row['BOS'] in ('bullish', 'bearish') or row['CHoCH'] in ('bullish', 'bearish'):
            tag = None
            color = None
            va = 'bottom'
            if row['BOS'] == 'bullish':
                tag, color = "BOS", COLORS['bos']
            elif row['BOS'] == 'bearish':
                tag, color = "BOS", COLORS['bos']
                va = 'top'
            elif row['CHoCH'] == 'bullish':
                tag, color = "CHoCH", COLORS['choch']
            elif row['CHoCH'] == 'bearish':
                tag, color = "CHoCH", COLORS['choch']
                va = 'top'
            if tag:
                ax.axvline(x=xpos, color=color, linestyle='-' if tag == 'BOS' else '--', linewidth=2, alpha=0.8)
                ax.text(xpos, row['close'], tag, color=color, fontsize=11, fontweight='bold', ha='center', va=va, zorder=12)
    swing_high_idx = df['Pivot_High'].dropna().index[-5:]
    swing_low_idx  = df['Pivot_Low'].dropna().index[-5:]
    int_high_idx = df['Internal_High'].dropna().index[-10:]
    int_low_idx  = df['Internal_Low'].dropna().index[-10:]
    ax.scatter([mdates.date2num(x) for x in swing_high_idx], df.loc[swing_high_idx, 'Pivot_High'], color=COLORS['swing_bear'], s=100, marker='v', zorder=8, label='Swing High')
    ax.scatter([mdates.date2num(x) for x in swing_low_idx], df.loc[swing_low_idx, 'Pivot_Low'], color=COLORS['swing_bull'], s=100, marker='^', zorder=8, label='Swing Low')
    ax.scatter([mdates.date2num(x) for x in int_high_idx], df.loc[int_high_idx, 'Internal_High'], color=COLORS['internal_bear'], s=55, marker='v', alpha=0.55, zorder=7)
    ax.scatter([mdates.date2num(x) for x in int_low_idx],  df.loc[int_low_idx,  'Internal_Low'],  color=COLORS['internal_bull'], s=55, marker='^', alpha=0.55, zorder=7)
    eqh = df['EQH'].dropna().tail(3)
    eql = df['EQL'].dropna().tail(3)
    for i, y in eqh.items():
        left = mdates.date2num(df.index[max(0, df.index.get_loc(i) - 3)])
        right = mdates.date2num(i)
        ax.hlines(y, xmin=left, xmax=right, colors=COLORS['eqh'], linestyles=':', linewidth=2)
        ax.text(right, y, 'EQH', color=COLORS['eqh'], fontsize=10, ha='left', va='bottom')
    for i, y in eql.items():
        left = mdates.date2num(df.index[max(0, df.index.get_loc(i) - 3)])
        right = mdates.date2num(i)
        ax.hlines(y, xmin=left, xmax=right, colors=COLORS['eql'], linestyles=':', linewidth=2)
        ax.text(right, y, 'EQL', color=COLORS['eql'], fontsize=10, ha='left', va='top')
    for block in ob_blocks[-3:]:
        idx = block['idx']
        width = 2
        if idx >= len(df) - 1: continue
        x = mdates.date2num(idx_list[idx])
        idx2 = min(idx + width, len(df) - 1)
        x2 = mdates.date2num(idx_list[idx2])
        rect_width = x2 - x
        if block['bull']:
            rect = Rectangle((x, block['bottom']), rect_width, block['top'] - block['bottom'],
                             color=COLORS['ob_bull'], alpha=0.15, linewidth=1, zorder=2)
            ax.add_patch(rect)
        else:
            rect = Rectangle((x, block['bottom']), rect_width, block['top'] - block['bottom'],
                             color=COLORS['ob_bear'], alpha=0.15, linewidth=1, zorder=2)
            ax.add_patch(rect)
    for gap in fvg_blocks[-3:]:
        idx1 = gap['start']
        idx2 = gap['end']
        if idx1 >= len(df) or idx2 >= len(df): continue
        x1 = mdates.date2num(idx_list[idx1])
        x2 = mdates.date2num(idx_list[idx2])
        rect_width = x2 - x1
        ytop = gap['top']
        ybot = gap['bottom']
        color = COLORS['fvg_bull'] if gap['bull'] else COLORS['fvg_bear']
        rect = Rectangle((x1, ybot), rect_width, ytop - ybot, color=color, alpha=0.07, linewidth=1.5, zorder=1)
        ax.add_patch(rect)
    for i in df.tail(2).index:
        y_top = df.loc[i, 'prem']
        y_bot = df.loc[i, 'disc']
        y_eq  = df.loc[i, 'eq']
        if not (np.isnan(y_top) or np.isnan(y_bot)):
            ax.axhspan(y_top, y_top*0.95+y_bot*0.05, color=COLORS['premium'], alpha=0.09)
            ax.axhspan(y_bot*0.95+y_top*0.05, y_bot, color=COLORS['discount'], alpha=0.09)
            ax.axhspan(y_bot*0.525+y_top*0.475, y_top*0.525+y_bot*0.475, color=COLORS['eq'], alpha=0.09)
    df['day'] = [x.date() for x in idx_list]
    if len(df['day'].unique()) > 0:
        last_day = df['day'].iloc[-1]
        subdf = df[df['day'] == last_day]
        dh = subdf['high'].max()
        dl = subdf['low'].min()
        sub_idx = subdf.index.to_list()
        left = mdates.date2num(sub_idx[0])
        right = mdates.date2num(sub_idx[-1])
        ax.hlines(dh, left, right, color=COLORS['daily'], linestyle='-.', linewidth=1, alpha=0.3)
        ax.hlines(dl, left, right, color=COLORS['daily'], linestyle='-.', linewidth=1, alpha=0.3)
    ax.set_ylabel("Price")
    ax.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

async def send_telegram_message(token, chat_id, text, photo_bytes=None):
    bot = Bot(token=token)
    try:
        if photo_bytes:
            img_data = photo_bytes.getvalue()
            photo = BufferedInputFile(img_data, filename="smc.png")
            await bot.send_photo(chat_id, photo=photo, caption=text)
        else:
            await bot.send_message(chat_id, text)
    finally:
        await bot.session.close()

async def analyze_all_pairs():
    exchange = ccxt.bybit({"enableRateLimit": True})
    markets = exchange.fetch_markets()
    pairs = [m['symbol'] for m in markets if m.get('contract') and m.get('settle') == 'USDT' and m.get('expiry') is None and m.get('active')]
    print(f"Парам для анализа: {len(pairs)}")
    for pair in pairs:
        for tf in TIMEFRAMES:
            try:
                ohlcv = exchange.fetch_ohlcv(pair, timeframe=tf, limit=HISTORY_LIMIT)
                if not ohlcv or len(ohlcv) < MIN_BARS:
                    print(f"[{pair}-{tf}] bars: {len(ohlcv)} — мало данных")
                    continue
                df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('time', inplace=True)
                signal = ""
                if 'Signal' in df.columns and df['Signal'].iloc[-1] in ("buy", "sell"):
                    signal = df['Signal'].iloc[-1].upper()
                    msg = f"SMC сигнал {signal} на {pair} ({tf}) {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
                else:
                    msg = f"SMC (визуализация) {pair} ({tf}) {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
                img_buf = plot_smc_full(df, pair, tf, signal)
                await send_telegram_message(API_TOKEN, TG_CHAT_ID, msg, photo_bytes=img_buf)
            except Exception as e:
                print(f"[ERROR] {pair}-{tf}: {e}")

if __name__ == "__main__":
    asyncio.run(analyze_all_pairs())
