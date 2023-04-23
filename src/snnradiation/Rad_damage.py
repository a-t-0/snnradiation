"""Contains the radiation settings supported for the SNNs."""

import hashlib
import json
from typing import List, Tuple, Union

from typeguard import typechecked


# pylint: disable=R0903
# pylint: disable=R0913
class Rad_damage:
    """Specification of a simulated radiation model: where neuron currents can
    decrease/increase randomly (if they are excited by a incoming
    radiation.)"""

    @typechecked
    def __init__(
        self,
        amplitude: float,
        effect_type: str,
        excitatory: bool,
        inhibitory: bool,
        probability_per_t: float,
    ) -> None:
        self.amplitude: float = amplitude
        self.effect_type: str = effect_type
        self.excitatory: bool = excitatory
        self.inhibitory: bool = inhibitory

        # Verify probability is within range.
        if probability_per_t > 1:
            raise ValueError(
                "Error, radiation effect probability can't be larger than 1."
                f" Found:{probability_per_t}"
            )
        if probability_per_t < 0:
            raise ValueError(
                "Error, radiation effect probability can't be smaller than 0."
                f" Found:{probability_per_t}"
            )
        self.probability_per_t: float = probability_per_t

        if effect_type not in [
            "change_u",
            "neuron_death",
            "rand_neuron_spike",
            "rand_synapse_spike",
        ]:
            raise NotImplementedError(f"Error, {effect_type} not implemented.")

    @typechecked
    def get_rad_settings_hash(self) -> str:
        """Converts radiation settings into a hash."""
        rad_settings: List[Tuple[str, Union[float, bool, str]]] = []
        for key, value in self.__dict__.items():
            rad_settings.append((key, value))
        rad_settings_hash: str = str(
            hashlib.sha256(
                json.dumps(sorted(rad_settings)).encode("utf-8")
            ).hexdigest()
        )
        return rad_settings_hash

    @typechecked
    def get_rad_hash(self, neuron_names: List[str], seed: int) -> str:
        """Return a deterministic hash of the radiation based on a list of
        neuron names."""
        neuron_names.append(self.get_rad_settings_hash())
        neuron_names.append(str(seed))

        rad_affected_neurons_hash: str = str(
            hashlib.sha256(
                json.dumps(neuron_names).encode("utf-8")
            ).hexdigest()
        )
        return rad_affected_neurons_hash