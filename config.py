from dotenv import load_dotenv
import os
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY')
PAIRS = [p.strip() for p in os.getenv('PAIRS','EUR/USD,GBP/USD,USD/JPY,BTC/USD').split(',')]

# M1 Configuration
TIMEFRAME_M1 = os.getenv('TIMEFRAME_M1','1min')
SIGNALS_PER_SESSION = int(os.getenv('SIGNALS_PER_SESSION','8'))

# Legacy M5 config (keep for compatibility)
TIMEFRAME_M5 = os.getenv('TIMEFRAME_M5','5min')
SIGNALS_PER_DAY = int(os.getenv('SIGNALS_PER_DAY','40'))
START_HOUR_UTC = int(os.getenv('START_HOUR_UTC','9'))
END_HOUR_UTC = int(os.getenv('END_HOUR_UTC','21'))

# Database
DB_URL = os.getenv('DB_URL','sqlite:///signals_optimized.db')

# Other
GAP_MIN_BEFORE_ENTRY = int(os.getenv('GAP_MIN_BEFORE_ENTRY','1'))
GALE_INTERVAL_MIN = int(os.getenv('GALE_INTERVAL_MIN','1'))
BEST_PARAMS_FILE = os.getenv('BEST_PARAMS_FILE','best_params.json')

# Walk-forward (legacy)
WALK_TOTAL_DAYS = int(os.getenv('WALK_TOTAL_DAYS', '20'))   
WALK_DAYS_WINDOW = int(os.getenv('WALK_DAYS_WINDOW', '5'))
WALK_DAYS_TEST = int(os.getenv('WALK_DAYS_TEST', '10'))
