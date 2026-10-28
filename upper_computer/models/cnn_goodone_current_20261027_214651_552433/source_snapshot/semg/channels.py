"""Physical ADC identities; never substitute zeros for a missing sensor."""
LABELS = {0: 'CH0 · PA7 · 喉咙', 1: 'CH1 · PC5 · 脸颊/下嘴角'}


def validate_channels(value=None):
    value = [0] if value is None else value
    if (not isinstance(value, list) or not value or
            any(type(c) is not int or c not in LABELS for c in value) or
            value != sorted(set(value))):
        raise ValueError('通道应为CH0、CH1或两者，按物理通道顺序且不能重复')
    return list(value)


def recording_channels(meta):
    return validate_channels(meta.get('selected_channels', [0]))


def require_channels(meta, channels):
    active = recording_channels(meta)
    if not set(channels).issubset(active):
        raise ValueError(f'录制启用通道为{active}，模型要求{channels}；缺失通道不能补零或替换')


def description(channels):
    return ' + '.join(LABELS[c] for c in channels)
