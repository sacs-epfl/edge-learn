from enum import Enum


class LearningMode(Enum):
    HYBRID = "H"
    ONLY_DATA = "OD"
    ONLY_WEIGHTS = "OW"
    BASELINE = "B"
