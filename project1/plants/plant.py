# Plant class
class Plant:
    def forward(self, control_signal, disturbance):
        return NotImplementedError("Must be implemented by subclass.")