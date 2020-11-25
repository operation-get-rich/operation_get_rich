import os
import shutil
import time
from datetime import datetime


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

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)