import json
from utils import format_usd

with open('../state_polygon_01_03_minute_chart_bars.json') as f:
    state = json.load(f)

performance = state['performance']
deltas = []
ommitted_count = 0
capital = 25000
for the_date in performance:
    for symbol in performance[the_date]:
        if performance[the_date][symbol]['rsi_1m_before'] < 0:
            ommitted_count += 1
            continue
        deltas.append(performance[the_date][symbol]['delta'])
        delta_float = float(performance[the_date][symbol]['delta'][:-1])
        capital *= 1 + (delta_float / 100)

accuracy = sum(1 for d in deltas if d[0] != '-') / len(deltas)
print(f'Accuracy: {accuracy * 100}%')
print(f'Trade Count: {len(deltas)}')
print(f'Ommited Count: {ommitted_count}')
print(f'End Capital: {format_usd(capital)}')
