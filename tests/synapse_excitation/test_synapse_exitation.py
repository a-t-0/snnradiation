"""Performs tests that verify lava simulation produces the same results as the
networkx simulation."""
from __future__ import annotations

import unittest

import numpy as np
from simsnn.core.networks import Network
from simsnn.core.nodes import LIF, RandomSpiker
from simsnn.core.simulators import Simulator
from typeguard import typechecked


class Test_synapse_excitation(unittest.TestCase):
    """Verifies whether the synapse excitation of a random spiker is added
    directly into a receiving neuron, or whether it has a delay of 1
    timestep."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

        # Create simsnn network of 2 neurons.
        self.testnet: Simulator = create_sample_network_of_2_neurons()

    @typechecked
    def test_nominal_state(
        self,
    ) -> None:
        """Tests whether:

        0. the source neuron starts sending spikes at t=2,
        1. and that the first of these spikes is received at t=3
        2. That the the output neuron spikes at t=3.
        """
        sim_duration: int = 5
        self.testnet.run(sim_duration)  # Initialise default 2 neuron test net.

        for target_index, target_neuron in enumerate(
            self.testnet.raster.targets
        ):
            for t in range(0, sim_duration):
                print(
                    f"{t},{target_neuron.ID}  V="
                    + f"{self.testnet.multimeter.V[t][target_index]} : "
                    + f"{self.testnet.raster.spikes[t][target_index]}"
                )

        # Assert the input neuron spikes once at t=1.
        input_node_id: int = 0
        self.assertFalse(self.testnet.raster.spikes[0][input_node_id])
        self.assertTrue(self.testnet.raster.spikes[1][input_node_id])
        self.assertFalse(self.testnet.raster.spikes[2][input_node_id])
        self.assertFalse(self.testnet.raster.spikes[3][input_node_id])
        self.assertFalse(self.testnet.raster.spikes[4][input_node_id])

        # Assert the output neuron voltage is 0 until the input spike comes
        # in at t=2 (on timestep after the neuron spiked at t=1) and than stays
        # constant at V=3
        output_node_id: int = 1
        self.assertEqual(self.testnet.multimeter.V[0][output_node_id], 0)
        self.assertEqual(self.testnet.multimeter.V[1][output_node_id], 0)
        self.assertEqual(self.testnet.multimeter.V[2][output_node_id], 3)
        self.assertEqual(self.testnet.multimeter.V[3][output_node_id], 3)
        self.assertEqual(self.testnet.multimeter.V[4][output_node_id], 3)

    @typechecked
    def test_radiated_synapse_behaviour(
        self,
    ) -> None:
        """Tests whether the random synapse excitation arrives at the time of
        the spike.

        0. the source neuron starts sending spikes at t=2,
        1. and that the first of these spikes is received at t=3
        2. That the the output neuron spikes at t=3.
        """
        sim_duration: int = 5
        self.testnet.run(sim_duration)  # Initialise default 2 neuron test net.

        for target_index, target_neuron in enumerate(
            self.testnet.raster.targets
        ):
            for t in range(0, sim_duration):
                print(
                    f"{t},{target_neuron.ID}  V="
                    + f"{self.testnet.multimeter.V[t][target_index]} : "
                    + f"{self.testnet.raster.spikes[t][target_index]}"
                )

        # Assert the input neuron spikes once at t=1.
        input_node_id: int = 0
        self.assertFalse(self.testnet.raster.spikes[0][input_node_id])
        self.assertTrue(self.testnet.raster.spikes[1][input_node_id])
        self.assertFalse(self.testnet.raster.spikes[2][input_node_id])
        self.assertFalse(self.testnet.raster.spikes[3][input_node_id])
        self.assertFalse(self.testnet.raster.spikes[4][input_node_id])

        # Assert the output neuron voltage is 0 until the input spike comes
        # in at t=2 (on timestep after the neuron spiked at t=1) and than stays
        # constant at V=3
        output_node_id: int = 1
        self.assertEqual(self.testnet.multimeter.V[0][output_node_id], 0)
        self.assertEqual(self.testnet.multimeter.V[1][output_node_id], 0)
        self.assertEqual(self.testnet.multimeter.V[2][output_node_id], 3)
        self.assertEqual(self.testnet.multimeter.V[3][output_node_id], 3)
        self.assertEqual(self.testnet.multimeter.V[4][output_node_id], 3)


@typechecked
def add_simulated_random_synapse_excitation(
    *, left_neuron: LIF, probability: float, right_neuron: LIF, seed: int
) -> Simulator:
    """Includes two neurons:

     0. random_spiker neuron - spikes randomly with probability: probability.
     1. synapse_excitation neuron - spikes if it receives an input from the
     random_spiker neuron and from the input neuron.
    The delay between input and synapse is 0, and the delay between synapse
    and output neuron is 0. (To simulate +1 on synapse.)
    """
    # Create new neuron that randomly spikes.
    # The amplitude in the rand_spiking node is the voltage spike, not
    # the output synapse spike.
    rand_spiking_node = RandomSpiker(
        p=probability,
        amplitude=1,
        rng=np.random.default_rng(seed=seed),
    )
    print(left_neuron)
    print(right_neuron)
    print(rand_spiking_node)


@typechecked
def create_sample_network_of_2_neurons() -> Simulator:
    """Creates a default simsnn network consisting of 2 nodes which are
    programmed."""
    net: Network = Network()
    sim: Simulator = Simulator(net)

    # Create a LIF neuron, with a membrane voltage threshold of 1,
    # a post spike reset value of 0 and no voltage decay (m=1).
    input_neuron = net.createLIF(
        ID="input",
        bias=3,
        du=1.0,
        m=1.0,
        thr=4,
        V_reset=0,
        V_min=-1000,
        spike_only_if_thr_exceeded=True,
    )
    # Inhibit input neuron from spiking.
    net.createSynapse(
        pre=input_neuron,
        post=input_neuron,
        ID="input-input",
        w=-100,
        d=1,
    )

    output_neuron = net.createLIF(
        ID="output",
        bias=0,
        du=1.0,
        m=1.0,
        thr=3,
        V_reset=0,
        spike_only_if_thr_exceeded=True,
    )

    # Create a Synapse, between the programmed neuron and the LIF neuron,
    # with a voltage weight of 1 and a delay of 1.
    net.createSynapse(
        pre=input_neuron,
        post=output_neuron,
        ID="input-output",
        w=3,
        d=1,
    )

    # Add all neurons to the raster
    sim.raster.addTarget([input_neuron, output_neuron])
    # Add all neurons to the multimeter
    sim.multimeter.addTarget([input_neuron, output_neuron])
    return sim
