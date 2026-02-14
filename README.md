# Polymarket Bitcoin Trading Bot

A trading bot for Polymarket's 5-minute Bitcoin prediction markets.

## ⚠️ Disclaimer

This bot trades with real money. Use at your own risk. Start with paper trading mode to test strategies.

## Features

- Fetch 5-minute Bitcoin prediction markets from Polymarket
- Real-time price monitoring via WebSocket
- Configurable trading strategies
- Paper trading mode for testing without real money
- Trade execution via Polymarket CLOB API

## Setup

```bash
# Clone the repo
git clone https://github.com/TheMudkip/polymarket-bitcoin-bot.git
cd polymarket-bitcoin-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy config and set your keys
cp config.example.json config.json
# Edit config.json with your wallet private key and preferences
```

## Configuration

Edit `config.json`:

```json
{
  "private_key": "your_private_key_here",
  "paper_trading": true,
  "bet_size": 10,
  "threshold": 0.02
}
```

- `private_key`: Your wallet private key (without 0x prefix)
- `paper_trading`: Set to `true` to simulate trades
- `bet_size`: Amount to bet per trade (in USDC)
- `threshold`: Price threshold to trigger trades

## Usage

```bash
# Run the bot
python main.py

# Run with specific market
python main.py --market "bitcoin-5min"
```

## How It Works

The bot monitors Bitcoin prediction markets on Polymarket:
- Markets like "Will Bitcoin be above $X at [time]"
- 5-minute resolution markets
- Buys "Yes" when probability is low, sells when it rises

## Disclaimer

Trading prediction markets involves risk. This bot is for educational purposes. Always test with paper trading first.
