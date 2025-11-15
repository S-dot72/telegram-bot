-- -------------------------
-- Table: subscribers
-- -------------------------
CREATE TABLE IF NOT EXISTS subscribers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL UNIQUE, -- Telegram user ID
    username TEXT DEFAULT NULL,       -- optionnel
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- -------------------------
-- Table: signals
-- -------------------------
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,
    direction TEXT NOT NULL,
    reason TEXT,
    ts_enter TEXT NOT NULL,
    ts_send TEXT,
    confidence REAL,
    payload_json TEXT,
    result TEXT,
    timeframe INTEGER DEFAULT 5,
    gale_level INTEGER DEFAULT 0,
    max_gales INTEGER DEFAULT 2,
    winning_attempt TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS metadata (
  k TEXT PRIMARY KEY,
  v TEXT
);
