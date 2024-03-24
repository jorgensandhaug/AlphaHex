# Base controller class
class Controller:
    # def __init__(self, params):
    #     self.params = params

    def forward(self, params, error_history, i):
        raise NotImplementedError("Must be implemented by subclass.")

    def update_parameters(self, params, grad, learning_rate):
        raise NotImplementedError("Must be implemented by subclass.")