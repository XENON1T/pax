"""Neural network reconstruction algorithms"""
import numpy as np

from pax import plugin
from pax.datastructure import ReconstructedPosition


class PosRecNeuralNet(plugin.TransformPlugin):

    """Reconstruct S2 x,y positions from top pmt hit pattern using a single hidden layer feed-foward neural net

    See Alex Kish' thesis for details:
    http://www.physik.uzh.ch/groups/groupbaudis/darkmatter/theses/xenon/Kish_THESISelectronic.pdf
    """

    def startup(self):
        """ Initialize the neural net.
        """
        self.input_channels = np.array(self.config['channels_top'])
        self.nn_output_unit = self.config['nn_output_unit']

        self.nn = NeuralNet(n_inputs=len(self.input_channels),
                            n_hidden=self.config['hidden_layer_neurons'],
                            n_output=2,   # x, y
                            weights=self.config['weights'],
                            biases=self.config['biases'])

    def transform_event(self, event):
        """Reconstruct the position of S2s in an event.
        """

        # For every S2 peak found in the event
        for peak in event.S2s():

            input_areas = peak.area_per_channel[self.input_channels]

            # Run the neural net
            # Input is fraction of top area (see Xerawdp, PositionReconstruction.cpp, line 246)
            # Convert from neural net's units to pax units
            nn_output = self.nn.run(input_areas/np.sum(input_areas)) * self.nn_output_unit

            peak.reconstructed_positions.append(ReconstructedPosition({
                'x': nn_output[0],
                'y': nn_output[1],
                'z': 42,
                'algorithm': 'NeuralNet'}))

        # Return the event such that the next processor can work on it
        return event


class NeuralNet():
    """Single hidden layer feed-forward neural net
     - Input layer without activation function or bias
     - Hidden layer with atanh(sum + bias) activation function
     - Output layer with sum + bias (i.e. identity) activation function
    All neurons in a layer are connected to all neurons in the previous layer.
    """

    def __init__(self, n_inputs, n_hidden, n_output, weights, biases):

        # Boilerplate for storing args...
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.weights = np.array(weights)
        self.biases = np.array(biases)

        # Sanity checks
        if not len(biases) == n_hidden + n_output:
            raise ValueError("Each hidden and output neuron must have a bias!")
        if not len(weights) == n_inputs * n_hidden + n_hidden * n_output:
            raise ValueError("Invalid length of weights for totally connected neuron layers.")

    def run(self, input_values):
        """Return the neural net's output (numpy array of output neuron values) on the input_values"""
        assert len(input_values) == self.n_inputs

        # Input layer neurons do nothing

        # Run the hidden layer, apply tanh activation function
        hidden_values = self.run_layer(input_values,
                                       self.weights[:self.n_inputs * self.n_hidden])
        hidden_values = np.tanh(hidden_values + self.biases[:self.n_hidden])

        # Run the output layer, apply identity activation function (just add bias)
        output_values = self.run_layer(hidden_values, self.weights[self.n_inputs * self.n_hidden:])
        return output_values + self.biases[self.n_hidden:]

    def run_layer(self, input_values, weights):
        # Dendrite values: weighted inputs
        dendrite_values = np.tile(input_values, len(weights)/len(input_values)) * weights
        # Sum weighted inputs for each hidden layer neuron separately
        return np.sum(dendrite_values.reshape(-1, len(input_values)), axis=1)
