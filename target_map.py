def __get_target(target):
    labels = {
        'yes': 0,
        'no': 1,
        'up': 2,
        'down': 3,
        'left': 4,
        'right': 5,
        'on': 6,
        'off': 7,
        'stop': 8,
        'go': 9
    }
    if target in labels:
        return labels[target]
    return 10  # unknown