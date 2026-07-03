def suppress_amp_autocast_warning(func):
    import warnings
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated",
                category=FutureWarning,
            )
            return func(*args, **kwargs)

    return wrapper
