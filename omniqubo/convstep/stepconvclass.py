class StepConvAbs:
    def __init__(self) -> None:
        pass

    def interpret(self, sample):
        raise NotImplementedError()

    def convert(self, model):
        raise NotImplementedError()
