"""Neural network position reconstruction"""
import numpy as np
from pax import plugin, utils


class PosRecNeuralNet(plugin.PosRecPlugin):

    """Reconstruct S2 x,y positions from top pmt hit pattern using a feed-foward neural net

    See Alex Kish' thesis for details:
    http://www.physik.uzh.ch/groups/groupbaudis/darkmatter/theses/xenon/Kish_THESISelectronic.pdf
    """

    def startup(self):
        """ Initialize the neural net.
        """
        if self.config['pmt_0_is_fake']:
            self.input_channels = self.pmts[1:]
        else:
            self.input_channels = self.pmts
        self.nn_output_unit = self.config['nn_output_unit']

        # Possibly scale the input of the activation function by a supplied value (float)
        activation_scale = self.config['activation_function_scale']

        # Apply the activation function to the output layer (bool)
        output_layer_function = self.config['output_layer_function']

        # Load the file defining the structure (number of nodes per layer)
        # as well as the weights and biases of the neural network
        data = np.load(utils.data_file_name(self.config['neural_net_file']))

        self.nn = NeuralNet(structure=data['structure'],
                            weights=data['weights'],
                            biases=data['biases'],
                            activation_scale=activation_scale,
                            output_layer_function=output_layer_function)

        data.close()

    def reconstruct_position(self, peak):
        input_areas = peak.area_per_channel[self.input_channels]

        # Run the neural net
        # Input is fraction of top area (see Xerawdp, PositionReconstruction.cpp, line 246)
        # Convert from neural net's units to pax units
        return self.nn.run(input_areas/np.sum(input_areas)) * self.nn_output_unit


class NeuralNet():
    """Feed-forward neural net with an arbitrary number of hidden layers
     - Input layer without activation function or bias
     - Hidden layers with tanh(sum + bias) activation function
     - Output layer with sum + bias or tanh(sum + bias) activation function
    All neurons in a layer are connected to all neurons in the previous layer.
    """

    def __init__(self, structure, weights, biases, activation_scale, output_layer_function):
        self.n_inputs = structure[0]
        self.n_output = structure[-1]
        self.n_layers = len(structure)

        self.structure = np.array(structure)
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.activation_scale = activation_scale
        self.output_layer_function = output_layer_function

        # Calculate the number of connections in the network, per-layer
        self.n_connections_per_layer = [structure[i] * structure[i+1] for i in range(self.n_layers - 1)]
        # Create lists of the first weights and bias indices in each layer
        self.weight_indices = np.insert(np.cumsum(self.n_connections_per_layer), 0, 0)
        self.bias_indices = np.insert(np.cumsum(self.structure[1:]), 0, 0)

        # Sanity checks
        if not len(structure) > 2:
            raise ValueError("There are not enough layers in the network, need at least 2+1")
        if not len(biases) == np.sum(structure[1:]):
            raise ValueError("Each hidden and output neuron must have a bias!")
        if not len(weights) == np.sum(self.n_connections_per_layer):
            raise ValueError("Invalid length of weights for totally connected neuron layers.")

    def run(self, input_values):
        """Return the neural net's output (numpy array of output neuron values) on the input_values"""
        assert len(input_values) == self.n_inputs

        # Input layer neurons do nothing
        hidden_values = input_values

        # Run all hidden layers, apply tanh activation function
        for hidden_layer_i in range(self.n_layers - 2):
            # First calculate what range of weights and biases we have to provide
            wr, br = self.get_indices_range(hidden_layer_i)

            hidden_values = self.run_layer(hidden_values, self.weights[wr[0]:wr[1]])
            hidden_values = np.tanh((hidden_values + self.biases[br[0]:br[1]]) * self.activation_scale)

        # Run the output layer, apply activation function
        wr, br = self.get_indices_range(self.n_layers - 2)
        output_values = self.run_layer(hidden_values, self.weights[wr[0]:wr[1]])
        if self.output_layer_function:
            return np.tanh((output_values + self.biases[br[0]:br[1]]) * self.activation_scale)
        else:
            return output_values + self.biases[br[0]:br[1]]

    def run_layer(self, input_values, weights):
        weighted_inputs = np.tile(input_values, len(weights)/len(input_values)) * weights
        # Sum weighted inputs for each hidden layer neuron separately
        return np.sum(weighted_inputs.reshape(-1, len(input_values)), axis=1)

    def get_indices_range(self, layer_i):
        """Return the range of weights and biases to be used in this layer"""
        weight_range = (self.weight_indices[layer_i],
                        self.weight_indices[layer_i + 1])
        bias_range = (self.bias_indices[layer_i],
                      self.bias_indices[layer_i + 1])
        return weight_range, bias_range
