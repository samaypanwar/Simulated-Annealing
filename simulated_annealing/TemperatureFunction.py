
def additive(temperature, alpha: float = 0.95) -> float:
    return temperature - alpha

def geometric(temperature, alpha: float = 0.95) -> float:
    return temperature * alpha

def slow_decrease(temperature, alpha: float = 0.95) -> float:
    return temperature / (1 + alpha * temperature)
