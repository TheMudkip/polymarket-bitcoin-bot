#!/usr/bin/env python3
"""
Polymarket Bitcoin 5-Minute Trading Bot
AI-powered trading for BTC Up/Down markets with safety limits.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import requests

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"

def get_current_market_slug() -> str:
    """Calculate the current 5-min BTC market slug based on system time."""
    import calendar
    from datetime import datetime, timedelta, timezone
    
    now = datetime.now(timezone.utc)
    
    # Round DOWN to current 5-minute interval (the active market)
    minutes = now.minute
    seconds = now.second
    
    # Current 5-min window started at this minute
    current_5min = now - timedelta(minutes=minutes % 5, seconds=seconds)
    current_5min = current_5min.replace(second=0, microsecond=0)
    
    # Get the timestamp for this window (market ends at this time)
    timestamp = int(calendar.timegm(current_5min.timetuple()))
    
    logger.info(f"Current time: {now.strftime('%H:%M:%S UTC')}")
    logger.info(f"Current 5-min window started: {current_5min.strftime('%H:%M UTC')}")
    logger.info(f"Market slug: btc-updown-5m-{timestamp}")
    logger.info(f"URL: https://polymarket.com/event/btc-updown-5m-{timestamp}")
    
    return f"btc-updown-5m-{timestamp}"

# Default market for 5-min BTC (calculated dynamically)
DEFAULT_MARKET_SLUG = get_current_market_slug()


class Portfolio:
    """Track portfolio value and positions."""
    
    def __init__(self, initial_investment: float, stop_loss_pct: float = 0.95):
        self.initial_investment = initial_investment
        self.stop_loss_pct = stop_loss_pct
        self.cash = initial_investment
        self.positions = {}  # token_id -> {'amount': float, 'price': float}
        self.trade_history = []
        
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            pos['amount'] * pos['price'] 
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def pnl_pct(self) -> float:
        """Calculate P&L as percentage."""
        return (self.total_value / self.initial_investment - 1) * 100
    
    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed (stop loss not hit)."""
        return self.total_value >= self.initial_investment * self.stop_loss_pct
    
    def get_max_bet(self, max_bet_pct: float) -> float:
        """Get maximum bet size based on portfolio."""
        return self.cash * max_bet_pct
    
    def record_trade(self, action: str, token: str, amount: float, price: float):
        """Record a trade in history."""
        self.trade_history.append({
            'time': datetime.now().isoformat(),
            'action': action,
            'token': token[:20] + '...',
            'amount': amount,
            'price': price,
            'total': amount * price,
            'portfolio_value': self.total_value
        })
    
    def __str__(self) -> str:
        return (f"Portfolio: ${self.total_value:.2f} "
                f"(PnL: {self.pnl_pct:+.2f}%) "
                f"Cash: ${self.cash:.2f}")


class GeminiAI:
    """Gemini-powered decision maker."""
    
    # Model configurations - try Pro first, then Flash
    MODELS = [
        {"name": "gemini-2.5-pro", "url_template": "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"},
        {"name": "gemini-2.0-flash", "url_template": "https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"},
    ]
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.current_model_index = 0  # Start with first model (Pro)
        self.last_decision = None
        self.same_decision_count = 0
        self.cache_duration = 30  # Cache decisions for 30 seconds
        self.last_call_time = 0
    
    def get_current_url(self):
        """Get the URL for the current model."""
        model = self.MODELS[self.current_model_index]
        return model["url_template"].format(model=model["name"], api_key=self.api_key)
    
    def should_call_api(self) -> bool:
        """Rate limiting for free tier."""
        import time
        now = time.time()
        # Max 15 calls per minute on free tier (be conservative: 10/min)
        if now - self.last_call_time < 6:  # Wait 6 seconds between calls
            return False
        return True
    
    def decide(self, market_data: dict, portfolio: Portfolio) -> str:
        """Get trading decision from Gemini with caching."""
        
        prompt = f"""You are a trading expert for Polymarket's 5-minute Bitcoin prediction markets.

CURRENT MARKET SITUATION:
- Up token price: {market_data.get('up_price', 'N/A')} (probability of Up)
- Down token price: {market_data.get('down_price', 'N/A')} (probability of Down)
- Market volume: ${market_data.get('volume', 0):.2f}
- Market liquidity: ${market_data.get('liquidity', 0):.2f}
- Market question: {market_data.get('question', 'N/A')}

YOUR PORTFOLIO:
- Total value: ${portfolio.total_value:.2f}
- Cash available: ${portfolio.cash:.2f}
- Initial investment: ${portfolio.initial_investment:.2f}
- P&L: {portfolio.pnl_pct:+.2f}%
- Max bet allowed: ${portfolio.get_max_bet(0.1):.2f}

INSTRUCTIONS:
Based on the market data, decide whether to:
1. BUY_UP - Buy the "Up" token (betting Bitcoin will go up)
2. BUY_DOWN - Buy the "Down" token (betting Bitcoin will go down)  
3. NO_TRADE - Don't place a trade this round

SAFETY RULES:
- NEVER risk more than 10% of your portfolio on a single bet
- If prices are near 50/50 (uncertainty is high), prefer NO_TRADE
- Consider the spread - only trade if you have a strong conviction
- The 5-minute timeframe is very short-term - be cautious

Respond with ONLY one of these exact responses:
BUY_UP
BUY_DOWN
NO_TRADE

Your decision:"""

        # Try models in order (Pro first, then Flash)
        for model_index in range(len(self.MODELS)):
            self.current_model_index = model_index
            url = self.get_current_url()
            model_name = self.MODELS[model_index]["name"]
            
            try:
                response = requests.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.3,
                            "maxOutputTokens": 50
                        }
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                    
                    # Parse response
                    if 'BUY_UP' in text.upper():
                        logger.info(f"Gemini ({model_name}) decision: BUY_UP")
                        return 'BUY_UP'
                    elif 'BUY_DOWN' in text.upper():
                        logger.info(f"Gemini ({model_name}) decision: BUY_DOWN")
                        return 'BUY_DOWN'
                    else:
                        logger.info(f"Gemini ({model_name}) decision: NO_TRADE")
                        return 'NO_TRADE'
                elif response.status_code == 429:
                    # Rate limited - try next model
                    logger.warning(f"Gemini ({model_name}) rate limited, trying next model...")
                    continue
                else:
                    logger.error(f"Gemini API error ({model_name}): {response.status_code}")
                    if model_index < len(self.MODELS) - 1:
                        continue  # Try next model
                    raise Exception(f"Gemini API error: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error calling Gemini ({model_name}): {e}")
                if model_index < len(self.MODELS) - 1:
                    continue  # Try next model
                raise
        
        # All models failed
        raise Exception("All Gemini models failed")


class PolymarketBot:
    """Main trading bot."""
    
    def __init__(self, config: dict):
        self.config = config
        # Always use current market (not from config) to ensure we start with the active window
        self.market_slug = get_current_market_slug()
        logger.info(f"Starting with current market: {self.market_slug}")
        self.paper_trading = config.get('paper_trading', True)
        self.max_bet_pct = config.get('max_bet_pct', 0.1)
        
        # Initialize portfolio
        initial = config.get('initial_investment', 100)
        stop_loss = config.get('stop_loss_pct', 0.95)
        self.portfolio = Portfolio(initial, stop_loss)
        
        # Initialize AI
        gemini_key = config.get('gemini_api_key', '')
        if gemini_key:
            self.ai = GeminiAI(gemini_key)
        else:
            self.ai = None
            logger.warning("No Gemini API key - AI decisions disabled")
        
        # Market data cache
        self.market_data = {}
    
    def get_next_market_slug(self, current_slug: str) -> str:
        """Calculate next 5-min BTC market slug."""
        import calendar
        from datetime import datetime, timedelta
        
        # Extract timestamp from slug (e.g., btc-updown-5m-1771116300)
        if 'btc-updown-5m-' in current_slug:
            try:
                current_ts = int(current_slug.split('-')[-1])
                # Add 300 seconds (5 minutes)
                next_ts = current_ts + 300
                return f"btc-updown-5m-{next_ts}"
            except:
                pass
        
        # Fallback: calculate from current time
        now = datetime.utcnow()
        minutes = now.minute
        seconds = now.second
        # Round up to next 5-min
        next_5min = now + timedelta(seconds=(300 - ((minutes * 60 + seconds) % 300)))
        next_5min = next_5min.replace(second=0, microsecond=0)
        next_ts = int(calendar.timegm(next_5min.timetuple()))
        return f"btc-updown-5m-{next_ts}"
        
    def get_market_data(self) -> dict:
        """Fetch current market data."""
        try:
            # Get market info (with cache-busting timestamp)
            import time
            resp = requests.get(f"{GAMMA_API}/markets?slug={self.market_slug}&_={int(time.time())}")
            markets = resp.json()
            
            if not markets:
                logger.warning("No market found")
                return {}
            
            market = markets[0]
            
            # Parse token IDs - use the actual field from API
            import json
            token_ids = json.loads(market.get('clobTokenIds', '[]'))
            outcome_prices = json.loads(market.get('outcomePrices', '[]'))
            
            # Get best bid/ask for reference
            best_bid = float(market.get('bestBid', 0))
            best_ask = float(market.get('bestAsk', 0))
            
            # Use best bid/ask as primary (more reliable than outcomePrices which can be stale at 50.5%)
            if best_bid > 0 and best_ask > 0:
                up_price = (best_bid + best_ask) / 2
                logger.info(f"Prices from bestBid/Ask: Up={up_price*100:.1f}%, Down={(1-up_price)*100:.1f}% (bid={best_bid}, ask={best_ask})")
            elif len(outcome_prices) > 0:
                # Fallback to outcomePrices if no bid/ask
                up_price = float(outcome_prices[0])
                logger.info(f"Outcome prices from API: Up={up_price*100:.1f}%")
            else:
                up_price = 0.5
                logger.warning("No price data available, defaulting to 50%")
                
            down_price = 1 - up_price
            
            # Check if market is closed based on endDate (more reliable than API)
            end_date = market.get('endDate', '')
            from datetime import datetime, timezone
            is_closed_by_time = False
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    is_closed_by_time = now >= end_dt
                    logger.info(f"Market end: {end_dt}, now: {now}, closed_by_time: {is_closed_by_time}")
                except Exception as e:
                    logger.error(f"Error parsing endDate: {e}")
            
            is_closed = market.get('closed', True) or is_closed_by_time
            
            up_token = token_ids[0] if len(token_ids) > 0 else None
            down_token = token_ids[1] if len(token_ids) > 1 else None
            
            logger.info(f"Token IDs: up={up_token}, down={down_token}")
            logger.info(f"Prices: Up={up_price*100:.1f}% | Down={down_price*100:.1f}% (bestBid={best_bid}, bestAsk={best_ask})")
            
            return {
                'question': market.get('question', ''),
                'slug': market.get('slug', ''),
                'up_token': up_token,
                'down_token': down_token,
                'up_price': up_price,
                'down_price': down_price,
                'volume': float(market.get('volumeNum', 0)),
                'liquidity': float(market.get('liquidityNum', 0)),
                'active': market.get('active', False),
                'closed': is_closed,
                'end_date': end_date
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def get_current_prices(self, token_id: str) -> Optional[dict]:
        """Get current prices from order book."""
        try:
            resp = requests.get(f"{CLOB_HOST}/orderbook?token_id={token_id}")
            book = resp.json()
            
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            
            best_bid = float(bids[0]['price']) if bids else 0
            best_ask = float(asks[0]['price']) if asks else 1
            
            return {
                'bid': best_bid,
                'ask': best_ask,
                'mid': (best_bid + best_ask) / 2
            }
        except Exception as e:
            logger.error(f"Error getting prices: {e}")
            return None
    
    def execute_trade(self, action: str, token_id: str, price: float, amount: float) -> bool:
        """Execute a trade (or simulate in paper mode)."""
        
        if not self.portfolio.can_trade:
            logger.error("STOP LOSS HIT - Trading disabled")
            return False
        
        if action == 'NO_TRADE':
            return True
        
        # Calculate actual amount
        cost = amount * price
        if cost > self.portfolio.cash:
            logger.warning(f"Insufficient cash for trade: ${cost:.2f} > ${self.portfolio.cash:.2f}")
            return False
        
        if self.paper_trading:
            logger.info(f"[PAPER] {action}: ${cost:.2f} at ${price:.4f}")
            self.portfolio.cash -= cost
            self.portfolio.positions[token_id] = {
                'amount': amount,
                'price': price
            }
            self.portfolio.record_trade(action, token_id, amount, price)
            return True
        else:
            # Real trading would go here
            logger.warning("Real trading not implemented yet")
            return False
    
    async def run_cycle(self):
        """Run one trading cycle."""
        logger.info(f"=== Trading Cycle ===")
        logger.info(str(self.portfolio))
        
        # Check stop loss
        if not self.portfolio.can_trade:
            logger.error("STOP LOSS HIT - Stopping trading")
            return False
        
        # Get market data
        market_data = self.get_market_data()
        
        if not market_data:
            logger.warning("No market data available")
            return False
        
        if market_data.get('closed'):
            # Market is closed - try to advance to next timestamp
            logger.warning("Market is closed, advancing to next 5-min window...")
            next_slug = self.get_next_market_slug(self.market_slug)
            if next_slug:
                logger.info(f"Switching to: {next_slug}")
                self.market_slug = next_slug
                # Save to config
                self.config['market_slug'] = next_slug
                import json
                with open('config.json', 'w') as f:
                    json.dump(self.config, f, indent=2)
                return True
            else:
                logger.warning("No next market available")
                return False
        
        logger.info(f"Market: {market_data.get('question', '')[:50]}...")
        logger.info(f"  Up: ${market_data.get('up_price', 0):.4f} | Down: ${market_data.get('down_price', 0):.4f}")
        
        # Get AI decision if available
        if self.ai:
            # Try Gemini with fallback to heuristic
            try:
                decision = self.ai.decide(market_data, self.portfolio)
                logger.info(f"AI Decision: {decision}")
            except Exception as e:
                logger.warning(f"AI error, using heuristic: {e}")
                # Fallback heuristic ( widened from 42/58 to 40/60)
                up_price = market_data.get('up_price', 0.5)
                if up_price < 0.40:
                    decision = 'BUY_UP'
                elif up_price > 0.60:
                    decision = 'BUY_DOWN'
                else:
                    decision = 'NO_TRADE'
                logger.info(f"Heuristic Decision: {decision}")
        else:
            # Simple heuristic if no AI
            if market_data.get('up_price', 0.5) < 0.45:
                decision = 'BUY_UP'
            elif market_data.get('down_price', 0.5) < 0.45:
                decision = 'BUY_DOWN'
            else:
                decision = 'NO_TRADE'
            logger.info(f"Decision (heuristic): {decision}")
        
        # Execute trade
        if decision == 'BUY_UP':
            token_id = market_data['up_token']
            price = market_data['up_price']
            max_bet = self.portfolio.get_max_bet(self.max_bet_pct)
            # Calculate shares we can buy
            amount = max_bet / price if price > 0 else 0
            self.execute_trade('BUY_UP', token_id, price, amount)
            
        elif decision == 'BUY_DOWN':
            token_id = market_data['down_token']
            price = market_data['down_price']
            max_bet = self.portfolio.get_max_bet(self.max_bet_pct)
            amount = max_bet / price if price > 0 else 0
            self.execute_trade('BUY_DOWN', token_id, price, amount)
        
        logger.info(f"Cycle complete. {str(self.portfolio)}")
        return True
    
    async def run(self, loop: bool = False, interval: int = 30):
        """Run the bot."""
        logger.info("Starting Polymarket BTC 5-Min Trading Bot")
        
        if self.paper_trading:
            logger.info("⚠️  PAPER TRADING MODE - No real money at risk")
        else:
            logger.info("🔴 LIVE TRADING MODE - Real money at risk!")
        
        if self.portfolio.can_trade:
            logger.info(f"Initial investment: ${self.portfolio.initial_investment:.2f}")
            logger.info(f"Stop loss: {self.portfolio.stop_loss_pct*100:.0f}% (${self.portfolio.initial_investment * self.portfolio.stop_loss_pct:.2f})")
        
        if loop:
            while True:
                await self.run_cycle()
                logger.info(f"Waiting {interval}s before next cycle...")
                await asyncio.sleep(interval)
        else:
            await self.run_cycle()


def load_config() -> dict:
    """Load config from file."""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        # Create default config
        config = {
            "gemini_api_key": "YOUR_GEMINI_API_KEY",
            "private_key": "",
            "initial_investment": 100,
            "paper_trading": True,
            "max_bet_pct": 0.1,
            "stop_loss_pct": 0.95,
            "market_slug": DEFAULT_MARKET_SLUG,
            "log_level": "INFO"
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created default config: {config_path}")
        logger.info("Please edit config.json with your API keys")
        return None
    
    with open(config_path) as f:
        return json.load(f)


async def main():
    parser = argparse.ArgumentParser(description='Polymarket BTC Trading Bot')
    parser.add_argument('--loop', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=30, help='Seconds between cycles')
    args = parser.parse_args()
    
    config = load_config()
    if not config:
        return
    
    # Validate config
    if config.get('gemini_api_key') == 'YOUR_GEMINI_API_KEY':
        logger.warning("Please set your Gemini API key in config.json")
    
    bot = PolymarketBot(config)
    await bot.run(loop=args.loop, interval=args.interval)


if __name__ == "__main__":
    asyncio.run(main())
