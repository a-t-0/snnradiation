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
    def get_hash(self) -> str:
        """Converts radiation settings into a hash."""
        sorted_rad_settings_list: List[
            Tuple[str, Union[float, bool, str]]
        ] = []
        for sorted_key in sorted(self.__dict__.keys()):
            # rad_settings_list.append((key, value))
            sorted_rad_settings_list.append(self.__dict__[sorted_key])

        rad_settings_hash: str = str(
            hashlib.sha256(
                json.dumps(sorted_rad_settings_list).encode("utf-8")
            ).hexdigest()
        )
        return rad_settings_hash

    @typechecked
    def get_rad_hash(self, neuron_names: List[str], seed: int) -> str:
        """Return a deterministic hash of the radiation based on a list of
        neuron names."""
        neuron_names.append(self.get_hash())
        neuron_names.append(str(seed))
        rad_affected_neurons_hash: str = str(
            hashlib.sha256(
                json.dumps(sorted(neuron_names)).encode("utf-8")
            ).hexdigest()
        )
        return rad_affected_neurons_hash

    @typechecked
    def get_filename(self) -> str:
        """Returns a filename string."""
        return (
            f"{self.effect_type}_"
            + f"ex:{self.excitatory}_"
            + f"in_h:{self.amplitude}_"
            + f"prob:{self.probability_per_t}_"
        )


def list_of_hashes_to_hash(hashes: List[str]) -> str:
    """Converts a list of hashes into a new hash."""
    return str(
        hashlib.sha256(json.dumps(sorted(hashes)).encode("utf-8")).hexdigest()
    )
