"""Performs tests that verify lava simulation produces the same results as the
networkx simulation."""
from __future__ import annotations

import unittest

import numpy as np
from simsnn.core.networks import Network
from simsnn.core.nodes import LIF, RandomSpiker
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snnradiation.apply_rad_to_simsnn import get_and_neuron


class Test_synapse_excitation(unittest.TestCase):
    """Verifies whether the synapse excitation of a random spiker is added
    directly into a receiving neuron, or whether it has a delay of 1
    timestep."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    @typechecked
    def test_nominal_state(
        self,
    ) -> None:
        """Tests whether:

        0. the source neuron starts sending spikes at t=2,
        1. and that the first of these spikes is received at t=3
        2. That the the output neuron spikes at t=3.
        """
        # Create simsnn network of 2 neurons.
        testnet: Simulator = create_sample_network_of_2_neurons()
        # Add all neurons to the raster
        testnet.raster.addTarget(testnet.network.nodes)
        # Add all neurons to the multimeter
        testnet.multimeter.addTarget(testnet.network.nodes)

        sim_duration: int = 5
        testnet.run(sim_duration)  # Initialise default 2 neuron test net.

        # Add raster and multimeter to monitor spike behaviour.

        for target_index, target_neuron in enumerate(testnet.raster.targets):
            for t in range(0, sim_duration):
                print(
                    f"{t},{target_neuron.ID}  V="
                    + f"{testnet.multimeter.V[t][target_index]} : "
                    + f"{testnet.raster.spikes[t][target_index]}"
                )

        # Assert the input neuron spikes once at t=1.
        input_node_id: int = 0
        self.assertFalse(testnet.raster.spikes[0][input_node_id])
        self.assertTrue(testnet.raster.spikes[1][input_node_id])
        self.assertFalse(testnet.raster.spikes[2][input_node_id])
        self.assertFalse(testnet.raster.spikes[3][input_node_id])
        self.assertFalse(testnet.raster.spikes[4][input_node_id])

        # Assert the output neuron voltage is 0 until the input spike comes
        # in at t=2 (on timestep after the neuron spiked at t=1) and than stays
        # constant at V=3
        output_node_id: int = 1
        self.assertEqual(testnet.multimeter.V[0][output_node_id], 0)
        self.assertEqual(testnet.multimeter.V[1][output_node_id], 0)
        self.assertEqual(testnet.multimeter.V[2][output_node_id], 3)
        self.assertEqual(testnet.multimeter.V[3][output_node_id], 3)
        self.assertEqual(testnet.multimeter.V[4][output_node_id], 3)

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
        # Create simsnn network of 2 neurons.
        testnet: Simulator = create_sample_network_of_2_neurons()

        add_simulated_random_synapse_excitation(
            left_neuron=testnet.network.nodes[0],
            right_neuron=testnet.network.nodes[1],
            sim=testnet,
            probability=0.99999,
            seed=1,
        )

        # Add all neurons to the raster
        testnet.raster.addTarget(testnet.network.nodes)
        # Add all neurons to the multimeter
        testnet.multimeter.addTarget(testnet.network.nodes)
        for i, node in enumerate(testnet.network.nodes):
            print(f"{i}:{node.ID}")

        sim_duration: int = 5
        testnet.run(sim_duration)  # Initialise default 2 neuron test net.

        for target_index, target_neuron in enumerate(testnet.raster.targets):
            for t in range(0, sim_duration):
                print(
                    f"{t},{target_neuron.ID}  V="
                    + f"{testnet.multimeter.V[t][target_index]} : "
                    + f"{testnet.raster.spikes[t][target_index]}"
                )

        # Assert the input neuron spikes once at for all timesteps.
        input_node_id: int = 0
        self.assertFalse(testnet.raster.spikes[0][input_node_id])
        self.assertTrue(testnet.raster.spikes[1][input_node_id])
        self.assertFalse(testnet.raster.spikes[2][input_node_id])
        self.assertFalse(testnet.raster.spikes[3][input_node_id])
        self.assertFalse(testnet.raster.spikes[4][input_node_id])

        # Assert the RandomSpiker neuron spikes once at for all timesteps.
        random_spiker_node_id: int = 2
        self.assertTrue(testnet.raster.spikes[0][random_spiker_node_id])
        self.assertTrue(testnet.raster.spikes[1][random_spiker_node_id])
        self.assertTrue(testnet.raster.spikes[2][random_spiker_node_id])
        self.assertTrue(testnet.raster.spikes[3][random_spiker_node_id])
        self.assertTrue(testnet.raster.spikes[4][random_spiker_node_id])

        # Assert the and neuron spikes at t=1.
        and_neuron_id: int = 3
        self.assertFalse(testnet.raster.spikes[0][and_neuron_id])
        self.assertFalse(testnet.raster.spikes[1][and_neuron_id])
        self.assertTrue(testnet.raster.spikes[2][and_neuron_id])
        self.assertFalse(testnet.raster.spikes[3][and_neuron_id])
        self.assertFalse(testnet.raster.spikes[4][and_neuron_id])

        # Assert the output neuron voltage is 0 until the input spike comes
        # in at t=2 (on timestep after the neuron spiked at t=1) and than stays
        # constant at V=3 +1 because of the and neuron input.
        output_node_id: int = 1
        self.assertEqual(testnet.multimeter.V[0][output_node_id], 0)
        self.assertEqual(testnet.multimeter.V[1][output_node_id], 0)
        self.assertEqual(testnet.multimeter.V[2][output_node_id], 3)
        self.assertEqual(testnet.multimeter.V[3][output_node_id], 4)
        self.assertEqual(testnet.multimeter.V[4][output_node_id], 4)


# @typechecked
def add_simulated_random_synapse_excitation(
    *,
    left_neuron: LIF,
    sim: Simulator,
    probability: float,
    right_neuron: LIF,
    seed: int,
) -> Simulator:
    """Includes two neurons:

     0. random_spiker neuron - spikes randomly with probability: probability.
     1. synapse_excitation neuron - spikes if it receives an input from the
     random_spiker neuron and from the input neuron.
    The delay between input and synapse is 0, and the delay between synapse
    and output neuron is 0. (To simulate +1 on synapse.)
    """
    net: Network = sim.network

    # Create new neuron that randomly spikes.
    # The amplitude in the rand_spiking node is the voltage spike, not
    # the output synapse spike.
    rand_spiking_node = RandomSpiker(
        ID="randomspiker",
        p=probability,
        amplitude=1,
        rng=np.random.default_rng(seed=seed),
    )
    net.nodes.append(rand_spiking_node)

    and_neuron: LIF = get_and_neuron(net=net)
    print(left_neuron)
    print(right_neuron)
    print(rand_spiking_node)

    net.createSynapse(
        pre=left_neuron,
        post=and_neuron,
        ID="left-and",
        w=1,
        d=0,
    )
    net.createSynapse(
        pre=rand_spiking_node,
        post=and_neuron,
        ID="rand-and",
        w=1,
        d=1,
    )

    net.createSynapse(
        pre=and_neuron,
        post=right_neuron,
        ID="and-output",
        w=1,
        d=0,
    )


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
        thr=5,
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
    return sim
