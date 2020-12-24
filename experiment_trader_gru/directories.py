import os

EXPERIMENT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(EXPERIMENT_ROOT_DIR, 'runs')
ALPACA_PAPER_TRADE_LOGS_DIR = os.path.join(EXPERIMENT_ROOT_DIR, 'alpaca_paper_trade_logs')