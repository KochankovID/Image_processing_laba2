import time


def get_time(f: callable, *args, **kwargs):
    start_time = time.perf_counter()
    result = f(*args, **kwargs)
    end_time = time.perf_counter()

    result_time = end_time - start_time
    print(f'Function {f.__name__} was working {result_time}')

    return result
