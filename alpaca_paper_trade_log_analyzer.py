import ast
import json


def get_log_for_date(the_date):
    global f, line, line_splitted, date_str
    with open('./alpaca_paper_trade.log') as f:
        for line in f:
            line_splitted = line.split()

            date_str = line_splitted[0]

            if date_str == the_date:
                with open('./alpaca_paper_trade_2020-12-11.log', 'a') as f2:
                    f2.write(line)


def get_trade_logs_by_time():
    global f, line, line_splitted, date_str
    payloads_by_time = {}
    with open('./alpaca_paper_trade_2020-12-11.log') as f:
        for line in f:
            line_splitted = line.split()

            date_str = line_splitted[0]
            hour_str = line_splitted[1]
            log_type = line_splitted[2]

            payload_start_index = len(date_str) + len(hour_str) + len(log_type) + 3
            try:
                payload = ast.literal_eval(line[payload_start_index:])
                payloads_by_time[date_str + hour_str] = payload
            except SyntaxError:
                print(f'Payload is not a dict {line}')
    trade_payloads_by_time = {t: p for t, p in payloads_by_time.items()
                              if p['type'] in ('trade_buy', 'trade_sell')}
    return trade_payloads_by_time


trade_logs = get_trade_logs_by_time()

trades = [trade_log['payload'] for trade_log in trade_logs.values()]

trades

# get_log_for_date('2020-12-11')
