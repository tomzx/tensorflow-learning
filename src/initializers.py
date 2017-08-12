import tensorflow as tf

class Initializer:
    def __call__(self, shape, dtype=tf.float32):
        raise NotImplemented()

class Zeros(Initializer):
    def __call__(self, shape, dtype=tf.float32):
        return tf.constant(0, shape=shape, dtype=dtype)

class Ones(Initializer):
    def __call__(self, shape, dtype=tf.float32):
        return tf.constant(1, shape=shape, dtype=dtype)

class Constant(Initializer):
    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=tf.float32):
        return tf.constant(self.value, shape=shape, dtype=dtype)

class RandomNormal(Initializer):
    def __init__(self, mean=0, stddev=0.5, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        return tf.random_normal(shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

class RandomUniform(Initializer):
    def __init__(self, minimum = -1, maximum = 1, seed=None):
        self.minimum = minimum
        self.maximum = maximum
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        return tf.random_uniform(shape, self.minimum, self.maximum, dtype=dtype, seed=self.seed)

# Aliases
zeros = Zeros
ones = Ones
constant = Constant
random_normal = RandomNormal
random_uniform = RandomUniform

def get(identifier):
    module_globals = globals()
    if identifier in module_globals and callable(module_globals[identifier]):
        return module_globals[identifier]()
    else:
        raise ValueError('Invalid initializer type {}' % identifier)