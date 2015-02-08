"""Neural network reconstruction algorithms"""
import numpy as np

from pax import plugin, units

from pax.datastructure import ReconstructedPosition


class PosRecNeuralNet(plugin.TransformPlugin):

    """Reconstruct S2 x,y positions using Alex Kish's / Xenon 100 neural net
    """

    def startup(self):
        """ Initialize the neural net.
        """
        self.nn = NeuralNet(
            n_inputs=98,
            n_hidden=30,
            n_output=2,
            weights=self.config['weights'],
            biases=self.config['biases'],
        )

    def transform_event(self, event):
        """Reconstruct the position of S2s in an event.
        """

        # For every S2 peak found in the event
        for peak in event.S2s():

            total_area_top = np.sum(peak.area_per_channel[1:98+1])

            # Run the neural net on pmt 1-98
            # Input is fraction of top area (see PositionReconstruction.cpp, line 246)
            # Convert from mm (Xenon100 units) to pax units
            nn_output = self.nn.run(peak.area_per_channel[1:98+1]/total_area_top) * units.mm

            peak.reconstructed_positions.append(ReconstructedPosition({
                'x': nn_output[0],
                'y': nn_output[1],
                'z': 42,
                'algorithm': 'X100NeuralNet'
            }))

        # Return the event such that the next processor can work on it
        return event


class NeuralNet():
    """Implementation of Alex Kish's Xenon100 neural net
     - Input layer without activation function
     - Hidden layer with atanh(sum + bias) activation function
     - Output layer with sum + bias (i.e. identity) activation functoin
    All neurons in a layer are connected to all neurons in the previous layer
    """

    def __init__(self, n_inputs, n_hidden, n_output, weights, biases):

        # Boilerplate for storing args...
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.weights = np.array(weights)
        self.biases = np.array(biases)

        # Sanity checks
        if not len(biases) == n_inputs + n_hidden + n_output:
            raise ValueError("Each neuron must have a bias!")
        if not len(weights) == n_inputs * n_hidden + n_hidden * n_output:
            raise ValueError("Invalid length of weights. I assumed all neurons are connected.")

    def run(self, input_values):
        assert len(input_values) == self.n_inputs

        # Input layer is not run -- input neuron biases are unused!

        # Run the hidden layer, apply tanh activation function
        hidden_values = self.run_layer(input_values,
                                       self.weights[:self.n_inputs * self.n_hidden])
        hidden_values = np.tanh(hidden_values + self.biases[self.n_inputs:self.n_inputs + self.n_hidden])

        # Run the output layer, apply identity activation function (just add bias)
        output_values = self.run_layer(hidden_values, self.weights[self.n_inputs * self.n_hidden:])
        return output_values + self.biases[self.n_inputs + self.n_hidden:]

    @staticmethod
    def run_layer(input_values, weights):
        dendrite_values = np.tile(input_values, len(weights)/len(input_values)) * weights
        return np.sum(dendrite_values.reshape(-1, len(input_values)), axis=1)
