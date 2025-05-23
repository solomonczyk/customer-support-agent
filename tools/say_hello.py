from tools import register

@register
def say_hello(name: str) -> str:
    return f"Привет, {name}!"
