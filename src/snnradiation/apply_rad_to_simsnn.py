"""Applies the radiation settings to the simsnn."""
from pprint import pprint
from typing import List

from simsnn.core.networks import Network
from simsnn.core.nodes import LIF, RandomSpiker
from simsnn.core.simulators import Simulator

from snnradiation.Rad_damage import Rad_damage


def apply_rad_to_simsnn(
    rad: Rad_damage,
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
    if rad.effect_type in ["change_u", "neuron_death"]:
        apply_delta_u_rad(
            rad=rad, snn=snn, ignored_neuron_names=ignored_neuron_names
        )
    elif rad.effect_type == "rand_neuron_spike":
        apply_rand_spiking_neuron_rad(
            rad=rad, snn=snn, ignored_neuron_names=ignored_neuron_names
        )
    elif rad.effect_type == "rand_synapse_spike":
        apply_rand_spiking_synapse_rad(
            rad=rad, snn=snn, ignored_neuron_names=ignored_neuron_names
        )


def apply_delta_u_rad(
    rad: Rad_damage, snn: Simulator, ignored_neuron_names: List[str]
) -> None:
    """Modifies the snn to apply a change in neuron currents to model simulated
    radiation effects.

    - Change in u is realised by adding 1 extra neuron per original neuron
    that:
        - feeds into the original neuron
        - spikes with synaptic weight = specified radiation amplitude.
        - spikes with specified probability (per timestep)
    """
    pprint(snn.__dict__)
    net: Network = snn.network
    for node in snn.network.nodes:
        if node.name not in ignored_neuron_names:
            # Create new neuron that randomly spikes.
            # neuron = net.createLIF(ID="ln", thr=1, V_reset=0, m=1)
            # The amplitude in the rand_spiking node is the voltage spike, not
            # the output synapse spike.
            rand_spiking_node = RandomSpiker(
                p=rad.probability_per_t, amplitude=1
            )
            net.nodes.append(rand_spiking_node)

            # Create new synapse into original neuron.
            net.createSynapse(
                pre=rand_spiking_node,
                post=node.name,
                w=rad.amplitude,
                d=1,
            )


def apply_rand_spiking_neuron_rad(
    rad: Rad_damage,
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
    for node in snn.network.nodes:
        if node.name not in ignored_neuron_names:
            # Create new neuron that randomly spikes.
            # The amplitude in the rand_spiking node is the voltage spike, not
            # the output synapse spike.
            rand_spiking_node = RandomSpiker(
                p=rad.probability_per_t, amplitude=1
            )
            net.nodes.append(rand_spiking_node)

            # Create new synapses into outgoing neighbours of original neuron.
            for synapse in net.synapses:
                if synapse.pre == node.name:
                    neighbour: LIF = synapse.post

                    net.createSynapse(
                        pre=rand_spiking_node,
                        post=neighbour,
                        w=synapse.w,
                        d=1,
                    )


def apply_rand_spiking_synapse_rad(
    rad: Rad_damage, snn: Simulator, ignored_neuron_names: List[str]
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
    for synapse in net.synapses:
        if synapse.pre.name not in ignored_neuron_names:
            # Create new neuron that randomly spikes.
            # The amplitude in the rand_spiking node is the voltage spike, not
            # the output synapse spike.
            rand_spiking_node = RandomSpiker(
                p=rad.probability_per_t, amplitude=1
            )
            net.nodes.append(rand_spiking_node)

            # Create new synapses into outgoing neighbours of original neuron.
            neighbour: LIF = synapse.post

            net.createSynapse(
                pre=rand_spiking_node,
                post=neighbour,
                w=synapse.w,
                d=1,
            )
