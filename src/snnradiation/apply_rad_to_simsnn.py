"""Applies the radiation settings to the simsnn."""
import copy
from typing import List, Tuple

import numpy as np
from simsnn.core.connections import Synapse, Synaptic_rad
from simsnn.core.networks import Network
from simsnn.core.nodes import LIF, RandomSpiker
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snnradiation.Rad_damage import Rad_damage


@typechecked
def apply_rad_to_simsnn(
    rad: Rad_damage,
    seed: int,
    snn: Simulator,
    ignored_neuron_names: List[str],
) -> None:
    """Modifies the snn to ensure the desired radiation effects are simulated.

    - Change in u is realised by adding 1 extra neuron per original neuron
    that:
        - feeds into the original neuron
        - spikes with synaptic weight = specified radiation amplitude.
        - spikes with specified probability (per timestep)

    - Random spiking neurons are realised by adding 1 extra neuron per
    original neuron that:
        - feeds into the the neighbours of the original neuron
        - spikes with synaptic weight equal to the original synapse weight.
        - spikes with specified probability (per timestep)

    - Random spiking synapses are realised by adding 1 extra neuron per synapse
    that:
        - feeds into the target neuron of that original synapse.
        - spikes with synaptic weight equal to the original synapse weight.
        - spikes with specified probability (per timestep).

    - Random neuron death is realised by adding 1 neuron per original neuron
    that:
        - feeds into the original neuron
        - spikes with -infinite weight as amplitude.
        - spikes with specified probability (per timestep).

    - Random synapse death is not implemented.
    """
    if rad.effect_type == "change_u":
        apply_delta_u_rad(
            rad=rad,
            seed=seed,
            snn=snn,
            ignored_neuron_names=ignored_neuron_names,
        )
    elif rad.effect_type == "rand_neuron_spike":
        apply_rand_spiking_neuron_rad(
            rad=rad,
            seed=seed,
            snn=snn,
            ignored_neuron_names=ignored_neuron_names,
        )
    elif rad.effect_type == "rand_synapse_spike":
        apply_rand_spiking_synapse_rad(
            rad=rad,
            seed=seed,
            snn=snn,
            ignored_neuron_names=ignored_neuron_names,
        )


@typechecked
def apply_delta_u_rad(
    rad: Rad_damage, seed: int, snn: Simulator, ignored_neuron_names: List[str]
) -> None:
    """Modifies the snn to apply a change in neuron currents to model simulated
    radiation effects.

    - Change in u is realised by adding 1 extra neuron per original neuron
    that:
        - feeds into the original neuron
        - spikes with synaptic weight = specified radiation amplitude.
        - spikes with specified probability (per timestep)
    """
    net: Network = snn.network
    new_nodes: List[Tuple[RandomSpiker, LIF]] = []

    # First get the new nodes that need to be added.
    for i, node in enumerate(snn.network.nodes):
        if node.name not in ignored_neuron_names:
            # Create new neuron that randomly spikes.
            # The amplitude in the rand_spiking node is the voltage spike, not
            # the output synapse spike.
            rand_spiking_node = RandomSpiker(
                p=rad.probability_per_t,
                amplitude=-10000,
                rng=np.random.default_rng(seed=seed + i),
            )
            new_nodes.append((rand_spiking_node, node))

    # Add the new nodes. Done in separate loop to prevent rand_spiking neurons
    # from getting rand_spiking_neurons.
    for rand_spiking_node, target_node in new_nodes:
        net.nodes.append(rand_spiking_node)
        # Create new synapse into original neuron.
        net.createSynapse(
            pre=rand_spiking_node,
            post=target_node,
            w=rad.amplitude,
            d=1,
        )


@typechecked
def apply_rand_spiking_neuron_rad(
    rad: Rad_damage,
    seed: int,
    snn: Simulator,
    ignored_neuron_names: List[str],
) -> None:
    """Modifies the snn to apply a random spiking neuron to model simulated
    radiation effects.

    - Random spiking neurons are realised by adding 1 extra neuron per
    original neuron that:
        - feeds into the the neighbours of the original neuron
        - spikes with synaptic weight equal to the original synapse weight.
        - spikes with specified probability (per timestep)
    """
    net: Network = snn.network
    new_nodes: List[Tuple[RandomSpiker, List[Tuple[LIF, float]]]] = []

    for i, node in enumerate(snn.network.nodes):
        if node.name not in ignored_neuron_names:
            # Create new neuron that randomly spikes.
            # The amplitude in the rand_spiking node is the voltage spike, not
            # the output synapse spike.
            rand_spiking_node = RandomSpiker(
                p=rad.probability_per_t,
                amplitude=1,
                rng=np.random.default_rng(seed=seed + i),
            )

            # Create new synapses into outgoing neighbours of original neuron.
            neighbour_synapses: List[Synapse] = []
            for synapse in net.synapses:
                if synapse.pre.name == node.name:
                    neighbour_synapses.append(synapse)

            new_nodes.append((rand_spiking_node, neighbour_synapses))

    for rand_spiking_node, synapses in new_nodes:
        net.nodes.append(rand_spiking_node)
        for synapse in synapses:
            net.createSynapse(
                pre=rand_spiking_node,
                post=synapse.post,
                w=synapse.w,
                d=1,
            )


@typechecked
def apply_rand_spiking_synapse_rad(
    rad: Rad_damage, seed: int, snn: Simulator, ignored_neuron_names: List[str]
) -> None:
    """Modifies the snn to apply a random spiking synapse to model simulated
    radiation effects.

    - Random spiking synapses are realised by adding 1 extra neuron per synapse
    that:
        - feeds into the target neuron of that original synapse.
        - spikes with synaptic weight equal to the original synapse weight.
        - spikes with specified probability (per timestep).
    """
    net: Network = snn.network
    new_synapses: List[Tuple[RandomSpiker, Synapse]] = []

    for i, synapse in enumerate(net.synapses):
        if synapse.pre.name not in ignored_neuron_names:
            # Create new neuron that randomly spikes.
            # The amplitude in the rand_spiking node is the voltage spike, not
            # the output synapse spike.
            rand_spiking_node = RandomSpiker(
                p=rad.probability_per_t,
                amplitude=1,
                rng=np.random.default_rng(seed=seed + i),
            )
            new_synapses.append((rand_spiking_node, synapse))

    for rand_spiking_node, synapse in new_synapses:
        net.nodes.append(rand_spiking_node)
        # Create new synapses into outgoing neighbours of original neuron.
        net.createSynapse(
            pre=rand_spiking_node,
            post=synapse.post,
            w=synapse.w,
            d=1,
        )


@typechecked
def apply_synapse_weight_increase_rad(
    *,
    est_sim_duration: int,
    ignored_neuron_names: List[str],
    rad: Rad_damage,
    seed: int,
    snn: Simulator,
) -> None:
    """Modifies the snn to swap the synapses out with the synapses with
    radiation."""
    if (
        rad.nr_of_synaptic_weight_increases is None
        and rad.probability_per_t is None
    ):
        raise ValueError(
            "Expected a specification of the nr of synaptic weight increases."
        )

    # Convert the radiation object into a Synaptic_rad object.
    synaptic_rad = Synaptic_rad(
        avg_weight_increase=rad.amplitude,
        effect_type=rad.effect_type,
        est_sim_duration=est_sim_duration,
        n_synapses=len(snn.network.synapses),
        neuron_names=list(map(lambda node: node.name, snn.network.nodes)),
        nswi=rad.nr_of_synaptic_weight_increases,
        probability_per_t=rad.probability_per_t,
        seed=seed,
    )

    copied_synapses = copy.deepcopy(snn.network.synapses)
    for count, synapse in enumerate(copied_synapses):
        if (
            synapse.pre.name not in ignored_neuron_names
            and synapse.post.name not in ignored_neuron_names
        ):
            # Replace the existing synapse with a radiated synapse.
            new_synapse: Synapse = Synapse(
                pre=synapse.pre,
                post=synapse.post,
                w=synapse.w,
                d=synapse.d,
                radiation=synaptic_rad,
            )
            # new_synapse.ID=snn.network.synapses[count].ID
            new_synapse.ID = count
            snn.network.synapses[count] = new_synapse


def get_and_neuron(*, net: Network) -> LIF:
    """Creates a neuron that spikes if it receives an input of 2."""
    and_neuron = net.createLIF(
        ID="and",
        bias=0,
        du=1.0,
        m=0.0,
        thr=2,
        V_reset=0,
    )
    return and_neuron
