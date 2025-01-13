# This function is adapted from transformers-interpret under the Apache License 2.0.
# Original source: https://github.com/cdpierse/transformers-interpret
# See the Apache License 2.0 at http://www.apache.org/licenses/LICENSE-2.0 for details.

from enum import Enum, unique


@unique
class AttributionType(Enum):
    INTEGRATED_GRADIENTS = "IG"
    INTEGRATED_GRADIENTS_NOISE_TUNNEL = "IGNT"


class NoiseTunnelType(Enum):
    SMOOTHGRAD = "smoothgrad"
    SMOOTHGRAD_SQUARED = "smoothgrad_sq"
    VARGRAD = "vargrad"
