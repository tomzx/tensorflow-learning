class Layer:
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.built = False

    def __call__(self, *args, **kwargs):
        inputs = args[0]

        input_shape = inputs.shape.as_list()

        if not self.built:
            self.build(input_shape)

        return self.call(inputs)

    def call(self, inputs):
        return inputs

    def build(self, input_shape):
        pass