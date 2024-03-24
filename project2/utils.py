import time

def timing_decorator(name, print_interval=1):
    """
    A decorator that measures the average execution time of the function
    over multiple calls and prints the average every 'print_interval' calls.
    """
    def decorator(func):
        total_time = 0
        total_calls = 0
        total_calls_since_last_print = 0
        total_time_since_last_print = 0

        def wrapper(*args, **kwargs):
            nonlocal total_time, total_calls, total_calls_since_last_print, total_time_since_last_print
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            total_time += end_time - start_time
            total_time_since_last_print += end_time - start_time
            total_calls += 1
            total_calls_since_last_print += 1
            average_time = total_time / total_calls

            if total_calls % print_interval == 0:
                average_time_since_last_print = total_time_since_last_print / total_calls_since_last_print
                print(f"Average execution time for the last {print_interval} executions of \033[92m{name}\033[0m: {average_time_since_last_print:.15f}. Total average after {total_calls} calls: {average_time:.15f} seconds")
                total_calls_since_last_print = 0
                total_time_since_last_print = 0

            return result

        return wrapper

    return decorator