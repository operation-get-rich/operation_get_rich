import signal
import time


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