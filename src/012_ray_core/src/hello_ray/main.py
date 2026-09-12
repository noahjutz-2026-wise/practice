import ray

@ray.remote
def square(x):
    return x ** 2


def main():
    ray.init()

    futures = [square.remote(i) for i in range(5)]

    print(ray.get(futures))
