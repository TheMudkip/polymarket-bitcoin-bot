# Polymarket Bitcoin 5-Minute Trading Bot

A trading bot for Polymarket's 5-minute Bitcoin prediction markets with AI-powered decision making.

## ⚠️ Disclaimer

This bot trades with real money. Use at your own risk. Start with paper trading mode to test strategies.

## Features

- 🤖 **Gemini AI Integration** - LLM decides buy/sell based on market analysis
- 🛡️ **Safety Limits** - Never loses more than initial investment
- 📊 **Portfolio Tracking** - Tracks all positions and P&L
- 📈 **Paper Trading Mode** - Test without real money
- 🎯 **Specific Market Focus** - Targets the 5-minute BTC Up/Down markets

## Setup

```bash
# Clone the repo
git clone https://github.com/TheMudkip/polymarket-bitcoin-bot.git
cd polymarket-bitcoin-bot

# Install dependencies
pip install -r requirements.txt

# Copy config and set your keys
cp config.example.json config.json
```

## Configuration

Edit `config.json`:

```json
{
  "gemini_api_key": "YOUR_GEMINI_API_KEY",
  "private_key": "your_wallet_private_key",
  "initial_investment": 100,
  "paper_trading": true,
  "max_bet_pct": 0.1,
  "stop_loss_pct": 0.95,
  "market_slug": "btc-updown-5m-1771113600",
  "log_level": "INFO"
}
```

### Config Options

| Option | Description |
|--------|-------------|
| `gemini_api_key` | Your Gemini API key for AI decision making |
| `private_key` | Wallet private key (without 0x) for real trading |
| `initial_investment` | Starting balance in USDC |
| `paper_trading` | Set `true` to simulate trades |
| `max_bet_pct` | Max % of portfolio to risk per bet (0.1 = 10%) |
| `stop_loss_pct` | Stop trading if portfolio drops below this % of initial (0.95 = 95%) |
| `market_slug` | The specific market to trade |

## Safety Features

1. **Stop Loss** - Bot stops trading if portfolio falls below `stop_loss_pct` of initial investment
2. **Max Bet Size** - Never bets more than `max_bet_pct` of portfolio
3. **Paper Trading Default** - Safe by default
4. **Position Limits** - Tracks total exposure

## How It Works

1. Fetches current prices for Up/Down tokens
2. Gets market data and context
3. Sends to Gemini AI for analysis
4. AI decides: BUY UP, BUY DOWN, or NO TRADE
5. Executes trade (or logs in paper mode)
6. Tracks portfolio value

## Gemini Prompt

The bot sends this context to Gemini:
- Current prices (Up/Down probabilities)
- Market volume and liquidity
- Recent price history
- Portfolio status

Gemini responds with one of:
- `BUY_UP` - Buy "Up" token
- `BUY_DOWN` - Buy "Down" token
- `NO_TRADE` - No action recommended

## Usage

```bash
# Run in paper trading mode
python bot.py

# Run continuously (for automation)
python bot.py --loop --interval 60
```

## Risk Warning

- This bot uses AI which may make poor decisions
- Past performance doesn't guarantee future results
- Only trade with money you can afford to lose
- Always test with paper trading first
