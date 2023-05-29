"""Tests whether the and neuron behaves as expected.."""
from __future__ import annotations

import itertools
import unittest

from simsnn.core.networks import Network
from simsnn.core.simulators import Simulator
from typeguard import typechecked


class Test_and_neuron(unittest.TestCase):
    """Verifies whether the synapse excitation of a random spiker is added
    directly into a receiving neuron, or whether it has a delay of 1
    timestep."""

    # Initialize test object
    @typechecked
    def __init__(self, *args, **kwargs) -> None:  # type:ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

    @typechecked
    def test_manual_and_neuron_behaviour(
        self,
    ) -> None:
        """Tests whether:

        0. the source neuron starts sending spikes at t=2,
        1. and that the first of these spikes is received at t=3
        2. That the the output neuron spikes at t=3.
        """

        left_train = [
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
        ]
        right_train = [
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
        ]
        expected = [
            "synaptic_delay_placeholder",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
        ]
        sim_duration: int = len(left_train)
        # Create simsnn network of 2 neurons.
        testnet: Simulator = create_sample_network_with_and_neuron(
            left_train=left_train,
            right_train=right_train,
        )
        for t in range(0, sim_duration):
            testnet.run(
                1, extend_multimeter=True, extend_raster=True
            )  # Initialise default 2 neuron test net.

            # Assert the output neuron spikes on timestep after
            # the input neurons spike simultaneously.
            and_neuron_id: int = 2
            print(
                f"t={t}, expected={expected[t+1]}, "
                + f"{testnet.network.nodes[and_neuron_id].__dict__}"
            )
            if t > 1:
                self.assertEqual(
                    testnet.network.nodes[and_neuron_id].out, expected[t]
                )

    @typechecked
    def test_nominal_and_neuron_behaviour(
        self,
    ) -> None:
        """Tests whether:

        0. the source neuron starts sending spikes at t=2,
        1. and that the first of these spikes is received at t=3
        2. That the the output neuron spikes at t=3.
        """

        # pylint: disable=R1702
        for sim_duration in range(2, 9):
            left_arrs: list = list(
                itertools.product([True, False], repeat=sim_duration)
            )
            right_arrs: list = list(
                itertools.product([True, False], repeat=sim_duration)
            )
            for left_str in left_arrs:
                for right_str in right_arrs:
                    # Create simsnn network of 2 neurons.
                    testnet: Simulator = create_sample_network_with_and_neuron(
                        left_train=[*left_str],
                        right_train=[*right_str],
                    )

                    for t in range(0, sim_duration):
                        testnet.run(
                            1, extend_multimeter=True, extend_raster=True
                        )  # Initialise default 2 neuron test net.

                        # Assert the output neuron spikes on timestep after
                        # the input neurons spike simultaneously.
                        and_neuron_id: int = 2
                        if t > 0:
                            if [*left_str][t - 1] == 1 and [*right_str][
                                t - 1
                            ] == 1:
                                self.assertEqual(
                                    testnet.network.nodes[and_neuron_id].out, 1
                                )
                            else:
                                self.assertEqual(
                                    testnet.network.nodes[and_neuron_id].out, 0
                                )


def create_sample_network_with_and_neuron(
    *, left_train: list, right_train: list
) -> Simulator:
    """Creates a default simsnn network consisting of 2 nodes which are fed
    into a third neuron that spikes if both of the input neurons spike at the
    same time."""
    net: Network = Network()
    sim: Simulator = Simulator(net)

    # Create a LIF neuron, with a membrane voltage threshold of 1,
    # a post spike reset value of 0 and no voltage decay (m=1).
    # left_train = net.createInputTrain(train=the_train, loop=False, ID="pn")
    left_train = net.createInputTrain(
        train=left_train, loop=False, ID="left_train"
    )

    right_train = net.createInputTrain(
        train=right_train, loop=False, ID="right_train"
    )
    # right_train = net.createInputTrain(train=the_train, loop=False, ID="pn")

    and_neuron = net.createLIF(
        ID="and",
        bias=0,
        du=1.0,
        m=0.0,
        thr=2,
        V_reset=0,
    )

    net.createSynapse(
        pre=left_train,
        post=and_neuron,
        ID="left_train-and",
        w=1,
        d=1,
    )
    net.createSynapse(
        pre=right_train,
        post=and_neuron,
        ID="right_train-and",
        w=1,
        d=1,
    )

    # Add all neurons to the raster
    sim.raster.addTarget([left_train, right_train, and_neuron])
    # Add all neurons to the multimeter
    sim.multimeter.addTarget([left_train, right_train, and_neuron])
    return sim
