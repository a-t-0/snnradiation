# Radiation Effect Simulation on Spiking Neural Network

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3106/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Code Coverage](https://codecov.io/gh/a-t-0/snn/branch/main/graph/badge.svg)](https://codecov.io/gh/a-t-0/snnalgorithms)

This applies different simulated radiation effects into incoming spiking neural
networks (SNNs).

## Parent Repository

These effects of the radiation, and respective SNN performance, can be analysed
using [this parent repository].
Together, these repos can be used to investigate the effectivity of various
[brain-adaptation] mechanisms applied to these algorithms, in order to increase
their \[radiation\] robustness. You can run it on various [backends], as well
as on a custom LIF-neuron simulator.

## Supported Radiation Types

Different forms of radiation effects may be simulated using this software. This
allows different chip-makers to analyse and improve the radiation robustness of
their respective SNNs based on the radiation interaction with the chip. Each
chip manufacturer is expected to derive their own radiation effects they
anticipate, based on the respective orbit, hardware type and shielding
materials. These effects can then be simulated on your own SNN of interest.

Feel free to send pull requests for compatibility with different:

- SNN algorithms
- hardware backends
- neural models
- synaptic models
- radiation effects

| Algorithm                            | Encoding | Adaptation | Radiation    |
| ------------------------------------ | -------- | ---------- | ------------ |
| Minimum Dominating Set Approximation | Sparse   | Redundancy | Neuron Death |
|                                      |          |            |              |
|                                      |          |            |              |

[backends]: https://github.com/a-t-0/snnbackends
[brain-adaptation]: https://github.com/a-t-0/snnadaptation
[this parent repository]: https://github.com/a-t-0/snncompare
