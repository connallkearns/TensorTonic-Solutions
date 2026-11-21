def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    grad = lambda x: 2 * a * x + b # Define the derivative
    update = lambda x: x - (lr * grad(x)) # Define the update function
    x = float(x0)
    for step in range(steps):
        x =  update(x) # Update x
    return x