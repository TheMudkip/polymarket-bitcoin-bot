#!/usr/bin/env python3
"""
Polymarket Bitcoin Trading Bot
Monitors 5-minute Bitcoin prediction markets and executes trades.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

import requests
import websockets
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()

# Polymarket API endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


class PolymarketClient:
    """Client for interacting with Polymarket APIs."""
    
    def __init__(self, config: dict):
        self.config = config
        self.paper_trading = config.get("paper_trading", True)
        self.bet_size = config.get("bet_size", 10)
        self.threshold = config.get("threshold", 0.02)
        
    def search_markets(self, query: str, limit: int = 10) -> list:
        """Search for markets by query."""
        url = f"{GAMMA_API}/markets"
        params = {
            "condition": query,
            "limit": limit,
            "active": True
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error searching markets: {e}")
            return []
    
    def get_market(self, slug: str) -> Optional[dict]:
        """Get market details by slug."""
        url = f"{GAMMA_API}/markets"
        params = {"slug": slug}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            markets = response.json()
            return markets[0] if markets else None
        except Exception as e:
            logger.error(f"Error getting market: {e}")
            return None
    
    def get_current_price(self, token_id: str) -> Optional[float]:
        """Get current price for a token."""
        url = f"{GAMMA_API}/prices"
        params = {"token_id": token_id}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            prices = response.json()
            # Handle the response - it might be a list or dict
            if isinstance(prices, list):
                for price in prices:
                    if price.get("token_id") == token_id:
                        return float(price.get("price", 0))
            elif isinstance(prices, dict):
                # Sometimes returns nested structure
                for key, value in prices.items():
                    if isinstance(value, list):
                        for price in value:
                            if price.get("token_id") == token_id:
                                return float(price.get("price", 0))
            return None
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None
    
    def get_market_token_id(self, market: dict) -> Optional[str]:
        """Extract token ID from market data."""
        # Try clobTokenIds first (newer API)
        clob_token_ids = market.get("clobTokenIds")
        if clob_token_ids:
            try:
                import json
                token_ids = json.loads(clob_token_ids)
                if token_ids and len(token_ids) > 0:
                    return token_ids[0]  # Return first token (Yes outcome)
            except:
                pass
        
        # Try tokens array (older format)
        tokens = market.get("tokens", [])
        if tokens and len(tokens) > 0:
            return tokens[0].get("token_id")
        
        return None
    
    def get_order_book(self, token_id: str) -> dict:
        """Get order book for a token."""
        url = f"{CLOB_HOST}/orderbook"
        params = {"token_id": token_id}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting order book: {e}")
            return {"bids": [], "asks": []}
    
    async def place_order(self, token_id: str, side: str, price: float, size: float) -> dict:
        """
        Place an order on Polymarket.
        Note: Only works with real trading. Paper trading just logs.
        """
        if self.paper_trading:
            logger.info(f"[PAPER] Would place order: {side} {size} {token_id} @ {price}")
            return {"success": True, "paper": True}
        
        # Real trading would use py_clob_client here
        # This requires a private key and wallet setup
        logger.warning("Real trading not implemented yet")
        return {"success": False, "error": "Real trading not implemented"}


class TradingBot:
    """Main trading bot logic."""
    
    def __init__(self, config: dict):
        self.client = PolymarketClient(config)
        self.config = config
        self.bitcoin_markets = []
        
    def find_bitcoin_markets(self):
        """Find Bitcoin prediction markets."""
        logger.info("Searching for Bitcoin markets...")
        
        # Search for 5-minute Bitcoin markets
        markets = self.client.search_markets("Bitcoin 5", limit=20)
        
        btc_markets = []
        for market in markets:
            question = market.get("question", "").lower()
            if "bitcoin" in question or "btc" in question:
                # Check if it's a 5-minute market
                if "5" in question or "min" in question:
                    btc_markets.append(market)
                    logger.info(f"Found market: {market.get('question')}")
        
        self.bitcoin_markets = btc_markets
        return btc_markets
    
    async def monitor_market(self, market: dict, duration: int = 60):
        """Monitor a market for price changes."""
        token_id = self.client.get_market_token_id(market)
        if not token_id:
            logger.warning(f"No token ID found for market: {market.get('question', 'Unknown')}")
            return
        
        question = market.get("question", "Unknown")
        logger.info(f"Monitoring: {question}")
        
        start_time = datetime.now()
        last_price = None
        
        while (datetime.now() - start_time).seconds < duration:
            current_price = self.client.get_current_price(token_id)
            
            if current_price is None:
                await asyncio.sleep(5)
                continue
            
            if last_price is not None:
                change = current_price - last_price
                if abs(change) > self.client.threshold:
                    logger.info(f"Price change detected: {last_price:.4f} -> {current_price:.4f} (Δ{change:+.4f})")
                    
                    # Trading logic
                    if change > self.client.threshold:
                        # Price went up significantly - could be a signal
                        logger.info(f"Potential buy signal at {current_price:.4f}")
                    elif change < -self.client.threshold:
                        logger.info(f"Price dropped at {current_price:.4f}")
            
            last_price = current_price
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def run(self):
        """Main bot loop."""
        logger.info("Starting Polymarket Bitcoin Trading Bot")
        
        if self.client.paper_trading:
            logger.info("Running in PAPER TRADING mode")
        
        # Find markets
        markets = self.find_bitcoin_markets()
        
        if not markets:
            logger.warning("No Bitcoin markets found")
            # Fall back to general Bitcoin markets
            all_markets = self.client.search_markets("Bitcoin", limit=10)
            logger.info(f"Found {len(all_markets)} general Bitcoin markets")
            for m in all_markets[:5]:
                logger.info(f"  - {m.get('question')}")
            return
        
        logger.info(f"Found {len(markets)} Bitcoin markets to monitor")
        
        # Monitor each market
        tasks = []
        for market in markets[:3]:  # Monitor up to 3 markets
            task = asyncio.create_task(self.monitor_market(market, duration=60))
            tasks.append(task)
        
        # Wait for all monitoring tasks
        await asyncio.gather(*tasks)


async def main():
    """Entry point."""
    # Load config
    config_path = "config.json"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        logger.info("Creating default config...")
        default_config = {
            "private_key": "",
            "paper_trading": True,
            "bet_size": 10,
            "threshold": 0.02,
            "log_level": "INFO"
        }
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        logger.info("Please edit config.json with your settings")
        return
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Run bot
    bot = TradingBot(config)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
