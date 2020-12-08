import json
import os
import time
from datetime import datetime
from shutil import copyfile

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S%z'
DATE_FORMAT = '%Y-%m-%d '
US_CENTRAL_TIMEZONE = 'US/Central'


def convert_pandas_timestamp_to_formatted_string(timestamp):
    return timestamp.strftime(DATETIME_FORMAT)


def get_all_ticker_names():
    ticker_names = []
    ticker_names += _get_ticker_names('stock_tickers.txt')
    return ticker_names


def _get_ticker_names(file_name):
    ticker_names = []
    exchange_file = open(file_name)
    lines = exchange_file.readlines()
    for line in lines:
        ticker_names.append(line.strip())
    return ticker_names


def create_file(path):
    if not os.path.exists(path):
        with open(path, 'w'):
            os.utime(path, None)
        return True
    return False


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_date(time_str):
    # type: (AnyStr) -> datetime
    datetime_obj = datetime.strptime(''.join(time_str.rsplit(':', 1)), '%Y-%m-%d %H:%M:%S%z')
    return datetime(year=datetime_obj.year,
                    month=datetime_obj.month,
                    day=datetime_obj.day,
                    tzinfo=datetime_obj.tzinfo
                    )


def get_date_time(time_str):
    # type: (AnyStr) -> datetime
    return datetime.strptime(''.join(time_str.rsplit(':', 1)), '%Y-%m-%d %H:%M:%S%z')


def get_date_string(time):
    """
    :param time: time string with format like  2019-10-17 09:30:00-04:00
    """
    return time[0:11]


def get_hour_string(time):
    """
    :param time: time string with format like  2019-10-17 09:30:00-04:00
    """
    return time[11:13]


def get_minute_string(time):
    """
    :param time: time string with format like  2019-10-17 09:30:00-04:00
    """
    return time[14:16]


def get_date_string_legacy(time):
    # type: (datetime) -> AnyStr
    return ' '.join(time.isoformat().split('T'))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(
                '%r  %2.2f ms' % \
                (method.__name__, (te - ts) * 1000)
            )
        return result

    return timed


def retry_download(method, retry_count=2, timeout=30):
    def time_out_handler(signum, frame):
        raise TimeoutError("Process taken too long")

    def _retry_download(*args, **kwargs):
        for _ in range(retry_count):
            try:
                signal.signal(signal.SIGALRM, time_out_handler)
                signal.alarm(timeout)
                return method(*args, **kwargs)
            except Exception as exc:
                print('Exception: ', exc)
                pass

    return _retry_download


def is_time2_greater(time1, time2):
    """
    :param time1: A date string with format like 2019-10-17 09:30:00-04:00
    :param time2: A date string with format like 2019-10-17 09:30:00-04:00
    """
    time1_date = get_date_string(time1).split('-')
    time2_date = get_date_string(time2).split('-')

    if int(time2_date[2]) > int(time1_date[2]):
        return True

    if int(time2_date[1]) > int(time1_date[1]):
        return True

    if int(time2_date[0]) > int(time1_date[0]):
        return True

    return False


def is_time_match(time, hour, minute):
    time_hour = int(get_hour_string(time))
    time_minute = int(get_minute_string(time))
    return (time_hour, time_minute) == (hour, minute)


def convert_gapped_up_stocks_directory_to_organized_by_date(
        gapped_up_stocks_dir,
        dest_dir
):
    """
    If the gapped up stocks directory is organized by stocks name,
    this funciton will create a new directory organized by dates, copying all the files in `gapped_up_stocks_dir`

    Example Params:
        gapped_up_stocks_dir = './alpaca_gaped_up_stocks_early_volume_1e5_gap_10'
        dest_dir = './alpaca_gaped_up_stocks_early_volume_1e5_gap_10_by_date'
    """
    for stock_dir in sorted(os.listdir(gapped_up_stocks_dir)):
        for stock_file in sorted(os.listdir(f'{gapped_up_stocks_dir}/{stock_dir}')):
            date = stock_file[-25:-15]
            create_dir(f'{dest_dir}/{date}')
            copyfile(f'{gapped_up_stocks_dir}/{stock_dir}/{stock_file}',
                     f'{dest_dir}/{date}/{stock_file}')


def get_current_filename(dunder_file):
    return os.path.basename(dunder_file)


def get_current_directory(dunder_file):
    return os.path.dirname(os.path.realpath(dunder_file))


def format_usd(capital):
    capital_formatted = '${:,.2f}'.format(capital)
    return capital_formatted


def update_state(state_file_location, update_func=None, *args, **kwargs):
    created = create_file(state_file_location)
    if created:
        state = {}
    else:
        with open(state_file_location, 'r') as state_file:
            state = json.load(state_file)

    if update_func:
        update_func(state, *args, **kwargs)

    if '_meta' in state:
        if 'first_state_update_call' not in state['_meta']:
            state['_meta']['first_state_update_call'] = datetime.datetime.now().strftime(DATETIME_FORMAT)
        state['_meta']['latest_state_update_call'] = datetime.datetime.now().strftime(DATETIME_FORMAT)

    with open(state_file_location, 'w') as state_file:
        json.dump(state, state_file)
