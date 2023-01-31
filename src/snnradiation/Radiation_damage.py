"""Simulates radiation damage on SNN.

Currently simulates it through simulating neuron death by setting the
threshold voltage to 1000 V (which will not be reached used in the
current MDSA approximation). TODO: include other radiation effects such
as "unexpected/random" changes in neuronal and synaptic properties.
"""
import random
from typing import List

import networkx as nx
from typeguard import typechecked


class Radiation_damage:
    """Creates expected properties of the spike_once neuron."""

    @typechecked
    def __init__(self, probability: float):
        self.neuron_death_probability = (
            probability  # % of neurons that will decay.
        )

    @typechecked
    def inject_simulated_radiation(
        self, get_degree: nx.DiGraph, probability: float, seed: int
    ) -> List[str]:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param probability:

        """
        # Get random neurons from list.
        dead_neuron_names = self.get_random_neurons(
            get_degree, probability, seed
        )

        store_dead_neuron_names_in_graph(
            G=get_degree, dead_neuron_names=dead_neuron_names
        )

        # Kill neurons.
        self.kill_neurons(get_degree, dead_neuron_names)

        return dead_neuron_names

    @typechecked
    def get_random_neurons(
        self, get_degree: nx.DiGraph, probability: float, seed: int
    ) -> List[str]:
        """

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param probability:
        :param adaptation_only:  (Default value = False)

        """

        # TODO: restore the probabilitiy  of firing instead of getting fraction
        # of neurons.
        nr_of_dead_neurons = int(len(get_degree) * probability)

        random.seed(seed)
        # Get a list of length nr_of_dead_neurons with random integers
        # These integers indicate which neurons die.

        rand_indices = random.sample(
            range(0, len(get_degree)), nr_of_dead_neurons
        )

        dead_neuron_names = []
        # TODO: fold instead of for.
        count = 0
        for node_name in get_degree:

            # for i,node_name in enumerate(get_degree):
            if count in rand_indices:
                dead_neuron_names.append(node_name)
            count = count + 1

        return dead_neuron_names

    @typechecked
    def kill_neurons(
        self, get_degree: nx.DiGraph, dead_node_names: List[str]
    ) -> None:
        """Simulates dead neurons by setting spiking voltage threshold to near
        infinity.

        Ensures neuron does not spike.

        :param get_degree: Graph with the MDSA SNN approximation solution.
        :param dead_node_names:
        """
        for node_name in dead_node_names:
            if node_name in dead_node_names:
                get_degree.nodes[node_name]["nx_lif"][0].vth.set(9999)


@typechecked
def store_dead_neuron_names_in_graph(
    *, G: nx.DiGraph, dead_neuron_names: List[str]
) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param dead_neuron_names:

    """

    for node_name in G.nodes:
        if node_name in dead_neuron_names:
            G.nodes[node_name]["rad_death"] = True
        else:
            G.nodes[node_name]["rad_death"] = False


@typechecked
def verify_radiation_is_applied(
    *, some_graph: nx.DiGraph, dead_neuron_names: List[str], rad_type: str
) -> None:
    """Goes through the dead neuron names, and verifies the radiation is
    applied correctly."""

    # TODO: include check to see if store_dead_neuron_names_in_graph is
    # executed correctly by checking whether the:
    # G.nodes[node_name]["rad_death"] = True
    if rad_type == "neuron_death":
        for node_name in some_graph:
            if node_name in dead_neuron_names:
                if not some_graph.nodes[node_name]["rad_death"]:
                    raise Exception(
                        'Error, G.nodes[node_name]["rad_death"] not set'
                    )
                if some_graph.nodes[node_name]["nx_lif"][0].vth.get() != 9999:
                    raise Exception(
                        "Error, radiation is not applied to:{node_name}, even"
                        + f" though it is in:{dead_neuron_names}"
                    )
    else:
        raise Exception(
            f"Error, radiation type: {rad_type} is not yet supported."
        )
