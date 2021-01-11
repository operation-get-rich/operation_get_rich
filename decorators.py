import signal
import time
from functools import wraps


# Created a custom Exception due to unexpected behavior when Ubuntu handle exception raised by Signal
class RetryTimeoutError(Exception):
    pass


def retry_with_timeout(retry_count=2, timeout=30):
    def decorator(method):
        def time_out_handler(signum, frame):
            raise RetryTimeoutError("Process taken too long")

        def _retry_download(*args, **kwargs):
            for _ in range(retry_count):
                try:
                    signal.signal(signal.SIGALRM, time_out_handler)
                    signal.alarm(timeout)
                    result = method(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result

        return wraps(method)(_retry_download)

    return decorator


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
