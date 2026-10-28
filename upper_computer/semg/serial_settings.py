"""Transport settings only; sampling and model contracts are independent."""
BAUDRATES = (115200, 230400)
DEFAULT_BAUDRATE = 230400


def validate_baudrate(value):
    if type(value) is not int or value not in BAUDRATES:
        raise ValueError('波特率仅支持 115200 或 230400。')
    return value
