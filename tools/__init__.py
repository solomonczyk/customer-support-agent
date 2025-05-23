registry = {}

def register(fn):
    registry[fn.__name__] = fn
    return fn
