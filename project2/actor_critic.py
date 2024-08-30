import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from hexnet import HexNet, SimpleHexNet
from utils import timing_decorator
from scipy.special import softmax
import onnxruntime as ort
import onnx
import io


from hex import Hex


# Wrapper for onnx inference session
class ONNXForwardFunction:
    def __init__(self, ort_session):
        self.ort_session = ort_session
        self.input_name = self.ort_session.get_inputs()[0].name

    def forward(self, x):
        return self.ort_session.run(None, {self.input_name: x})
    
    def __call__(self, x):
        return self.forward(x)


class ActorCritic():
    """
    Generic Actor-Critic class that can utilize any neural network architecture.
    """
    def __init__(self, net, optimizer_class=optim.Adam, optimizer_params={}, save_folder=None, device="cpu", size=None):
        self.net = net
        self.optimizer = optimizer_class(self.net.parameters(), **optimizer_params)

        # Save folder
        if save_folder is not None:
            self.save_folder = save_folder
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

        self.device = device

        assert size is not None, "size must be specified"
        self.size = size
        self.net_class = net.__class__
        self.use_onnx = False

    
    # @timing_decorator("ACNET state_to_torch", print_interval=1000)
    def state_to_torch(self, state):
        if self.net_class == HexNet:
            state = state.conv_representation()
            
        elif self.net_class == SimpleHexNet:
            state = state.nn_representation()

        else:
            raise Exception(f"Unknown net class: {self.net_class}")

        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)  # Add batch dimension
        if self.use_onnx:
            x = self.to_numpy(x)

        return x

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def example_net_input(self):
        # Return an example input to the network
        p1 = Hex.example_net_input(self.size)
        return self.state_to_torch(p1)

    def forward(self, state):
        return self.net(state)

    @timing_decorator("ANET train", print_interval=1000)
    def train(self, minibatch):
        states, action_probs, state_action_values = zip(*minibatch)
        states = np.array(states)
        action_probs = np.array(action_probs)
        state_action_values = np.array(state_action_values)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        action_probs = torch.tensor(action_probs, dtype=torch.float32, device=self.device)  # Action probs should be float not long, since we use CrossEntropyLoss with probabilities
        state_action_values = torch.tensor(state_action_values, dtype=torch.float32, device=self.device).unsqueeze(1)

        predicted_logits, predicted_state_action_values = self.forward(states)
        
        policy_loss = nn.CrossEntropyLoss()(predicted_logits, action_probs)
        value_loss = nn.MSELoss()(predicted_state_action_values, state_action_values)

        self.optimizer.zero_grad()

        # Weighting is a bit arbitrary, but it seems to work fine for training the network and both heads
        total_loss = 2 * policy_loss + value_loss # TODO: add regularization term
        total_loss.backward() 
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def save_weights(self, episode, final=False):
        if self.save_folder is None:
            raise Exception("Cannot save weights if save_folder is not specified!")

        filename = f"{self.save_folder}/anet_weights_{'final' if final else episode}.pth"
        torch.save(self.net.state_dict(), filename)
        print(f"Model weights saved to {filename}")

        # Save to ONNX
        onnx_filename = f"{self.save_folder}/anet_weights_{'final' if final else episode}.onnx"
        self.compile_model_onnx(save_path=onnx_filename)
        print(f"Model saved in ONNX format to {onnx_filename}")

        # close the file
        return filename

    
    # @timing_decorator("ANET get_action", print_interval=500)
    def get_action(self, state, epsilon=0.0):
        # Get the action based on the policy network
        legal_actions_mask = state.get_legal_actions_flat_mask()

        legal_indices = np.nonzero(legal_actions_mask)[0]

        if np.random.rand() < epsilon:
            # choose random item from legal_indices
            action_index = np.random.choice(legal_indices)
            action = state.action_index_to_action(action_index)
            return action


        with torch.no_grad():
            policy_logits, _ = self.forward(self.state_to_torch(state))  # Add batch dimension

        # Handle differently based on whether we are using ONNX or not
        if self.use_onnx:
            policy_probs = softmax(policy_logits, axis=1)
            policy_probs = policy_probs.squeeze(0)
        else:
            policy_probs = torch.softmax(policy_logits, dim=1)
            policy_probs = policy_probs.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

        # use legal_actions_mask as mask to remove illegal actions
        legal_action_probs = policy_probs * legal_actions_mask

        # choose action based on legal_action_probs
        action_index = np.argmax(legal_action_probs)

        action = state.action_index_to_action(action_index) 
        return action

    def get_value(self, state):
        # Get the value of the state
        with torch.no_grad():
            _, value = self.forward(self.state_to_torch(state))
        return value.squeeze(0).item()  # Remove batch dimension

    def get_action_and_probs(self, state):
        # Get the action based on the policy network and the probabilities of each action
        legal_actions_mask = state.get_legal_actions_flat_mask()

        legal_indices = np.nonzero(legal_actions_mask)[0]

        with torch.no_grad():
            policy_logits, _ = self.forward(self.state_to_torch(state))

        print(f"VALUE: {_}")
        print(f"policy_logits: \n{policy_logits}")

        if self.use_onnx:
            policy_probs = softmax(policy_logits, axis=1)
            policy_probs = policy_probs.squeeze(0)
        else:
            policy_probs = torch.softmax(policy_logits, dim=1)
            policy_probs = policy_probs.squeeze(0).cpu().numpy()
        print(f"policy_probs: \n{policy_probs}")

        # use legal_actions_mask as mask to remove illegal actions
        legal_action_probs = policy_probs * legal_actions_mask

        action_index = np.argmax(legal_action_probs)
        
        action = state.action_index_to_action(action_index)

        return action, policy_probs


    
    def get_action_from_best_2_based_on_probs(self, state):
        legal_actions_mask = state.get_legal_actions_flat_mask()

        with torch.no_grad():
            policy_logits, _ = self.forward(self.state_to_torch(state))
        
        if self.use_onnx:
            policy_probs = softmax(policy_logits, axis=1)
            policy_probs = policy_probs.squeeze(0)
        else:
            policy_probs = torch.softmax(policy_logits, dim=1)
            policy_probs = policy_probs.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy
        
        # use legal_actions_mask as mask to remove illegal actions
        legal_action_probs = policy_probs * legal_actions_mask
        
        # only keep the best 2 actions
        best_2_indices = np.argsort(legal_action_probs)[-2:]

        assert len(best_2_indices) == 2, f"best_2_indices: {best_2_indices}, legal_action_probs: {legal_action_probs}, policy_probs: {policy_probs}, legal_actions_mask: {legal_actions_mask}, policy_logits: {policy_logits}, state: {state}, self.state_to_torch(state): {self.state_to_torch(state)}"

        best_2_mask = np.zeros_like(legal_action_probs)
        best_2_mask[best_2_indices] = 1
        
        legal_action_probs = legal_action_probs * best_2_mask
        # rescale the probabilities to sum to 1
        legal_action_probs = legal_action_probs / np.sum(legal_action_probs)

        # choose action based on legal_action_probs
        action_index = np.random.choice(np.arange(len(legal_action_probs)), p=legal_action_probs)
        
        action = state.action_index_to_action(action_index)
        return action


    def __call__(self, state, epsilon=0.0):
        return self.get_action(state, epsilon=epsilon)

    def initialize_weights_from_file(self, filename):
        self.net.load_state_dict(torch.load(filename))
        self.use_onnx = False
        print(f"Initialized weights from {filename}")



    def compile_model_onnx(self, save_path=None):
        # An example input normally provided to the models's forward() method.
        example_input = self.example_net_input()

        # buffer object to store ONNX model
        onnx_model_buffer = io.BytesIO()

        # TODO: quantize the model before exporting to ONNX?

        # Export the model to ONNX format
        torch.onnx.export(self.net.eval(), example_input, onnx_model_buffer, opset_version=17,
        export_params=True,
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}},
        training=torch.onnx.TrainingMode.EVAL,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

        # export with dynamo
        # onnx_program = torch.onnx.dynamo_export(self.net.eval(), example_input)
        # onnx_program.save(onnx_model_buffer)

        # Get the serialized ONNX model
        model_serialized = onnx_model_buffer.getvalue()

        # Optionally save the model to a file
        if save_path is not None:
            with open(save_path, 'wb') as f:
                f.write(model_serialized)

        return model_serialized



    def compile_model_jit(self):
        # use torchscript and jit to compile the acnet for maximum inference speed

        # An example input normally provided to the models's forward() method.
        example_input = self.example_net_input()
        # print(f"Example input size: {example_input}")
        
        # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        model = torch.jit.trace(self.net.eval(), example_input)

        # Serialize the model
        model_buffer = io.BytesIO()
        torch.jit.save(model, model_buffer)
        model_serialized = model_buffer.getvalue()

        return model_serialized

    def initialize_net_from_jit(self, model_serialized):
        model = torch.jit.load(io.BytesIO(model_serialized))
        self.net_non_compiled = self.net
        self.net = model
        
        # test that the model works and gives the same output as the original model
        # self.test_jit_model_gives_same_output_as_original_model()

    def test_jit_model_gives_same_output_as_original_model(self):
        # Test that the JIT model gives the same output as the original model
        example_input = self.example_net_input()
        self.net.eval()
        self.net_non_compiled.eval()
        original_policy, original_value = self.net_non_compiled(example_input)
        jit_output_policy, jit_output_value = self.net(example_input)
        print(f"Original policy: {original_policy}, JIT policy: {jit_output_policy}")
        print(f"Original value: {original_value}, JIT value: {jit_output_value}")

        assert np.allclose(original_policy.detach().numpy(), jit_output_policy.detach().numpy(), atol=1e-6, rtol=1e-3), f"Original policy: {original_policy}, JIT policy: {jit_output_policy}"

    def initialize_net_from_onnx(self, model_serialized):
        # Create an inference session with ONNX runtime directly from the serialized model
        ort_session = ort.InferenceSession(model_serialized, providers=['CPUExecutionProvider'])

        self.use_onnx = True
        self.net_non_compiled = self.net
        self.net = ONNXForwardFunction(ort_session)


        # test that the model works and gives the same output as the original model
        # self.test_onnx_model_gives_same_output_as_original_model() #Commented this line due to the time it takes to run

    def test_onnx_model_gives_same_output_as_original_model(self):
        # Test that the ONNX model gives the same output as the original model
        self.use_onnx = False
        example_input = self.example_net_input()
        self.use_onnx = True
        self.net_non_compiled.eval()
        # check that it is in eval mode
        with torch.no_grad():
            original_policy, original_value = self.net_non_compiled(example_input)
            onnx_output_policy, onnx_output_value = self.net(self.to_numpy(example_input))
            print(f"Original policy: {original_policy}, ONNX policy: {onnx_output_policy}")
            print(f"Original value: {original_value}, ONNX value: {onnx_output_value}")


            assert np.allclose(original_policy.detach().numpy(), onnx_output_policy, atol=1e-6, rtol=1e-3), f"Original policy: {original_policy}, ONNX policy: {onnx_output_policy}"

    def use_uncompiled_network(self):
        # make sure the ONNX session is closed first
        if self.use_onnx:
            # Instantiate the original network from the saved torch model
            self.net = self.net_non_compiled
            self.use_onnx = False


    

# ACCNN class uses HexNet as its underlying model.
class ACCNN(ActorCritic):
    def __init__(self, size, in_channels, num_residual_blocks, num_filters, policy_output_dim, kernel_size, optimizer_class=optim.Adam, optimizer_params={}, save_folder=None, device="cpu"):
        net = HexNet(size, in_channels, num_residual_blocks, num_filters, policy_output_dim, kernel_size=kernel_size, device=device)
        super(ACCNN, self).__init__(net=net, optimizer_class=optimizer_class, optimizer_params=optimizer_params, save_folder=save_folder, device=device, size=size)
        

    def copy_and_initialize_weights_from_file(self, filename):
        # clone the acnet, then load the weights from filename
        copy = ACCNN(self.size, self.net.in_channels, self.net.num_residual_blocks, self.net.num_filters, self.net.policy_output_dim, self.net.kernel_size, device=self.net.device)
        copy.initialize_weights_from_file(filename)
        return copy
    
    def copy(self):
        # clone the acnet
        copy = ACCNN(self.size, self.net.in_channels, self.net.num_residual_blocks, self.net.num_filters, self.net.policy_output_dim, self.net.kernel_size, device=self.net.device)
        copy.net.load_state_dict(self.net.state_dict())
        return copy





# The ACNET class uses SimpleHexNet as its underlying model.
class ACNET(ActorCritic):
    def __init__(self, state_dim, action_dim, hidden_layers, activation_functions, optimizer_class=optim.Adam, optimizer_params={}, save_folder=None, device="cpu", size=None):
        net = SimpleHexNet(state_dim, action_dim, hidden_layers, activation_functions, device=device)
        super(ACNET, self).__init__(net=net, optimizer_class=optimizer_class, optimizer_params=optimizer_params, save_folder=save_folder, device=device, size=size)
        
    def copy_and_initialize_weights_from_file(self, filename):
        copy = ACNET(self.net.state_dim, self.net.action_dim, self.net.hidden_layers, self.net.activation_functions, device=self.net.device, size=self.size)
        copy.initialize_weights_from_file(filename)
        return copy
   
    def copy(self):
        copy = ACNET(self.net.state_dim, self.net.action_dim, self.net.hidden_layers, self.net.activation_functions, device=self.net.device, size=self.size)
        copy.net.load_state_dict(self.net.state_dict())
        return copy
