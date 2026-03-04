import random
from concurrent.futures import ThreadPoolExecutor

def monte_carlo_pi(num_samples):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1:
            inside_circle += 1
    return inside_circle

def parallel_monte_carlo(num_samples, num_threads):
    samples_per_thread = num_samples // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(monte_carlo_pi, [samples_per_thread] * num_threads)
    return sum(results)

num_samples = 100000000
num_threads = 8
inside_circle = parallel_monte_carlo(num_samples, num_threads)
pi_estimate = (inside_circle / num_samples) * 4
print(f"Оценка числа pi: {pi_estimate}")
