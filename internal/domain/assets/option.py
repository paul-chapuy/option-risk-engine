from enum import Enum, auto
from dataclasses import dataclass


class OptionType(Enum):
    Call = auto()
    Put = auto()


class ExcerciceStyle(Enum):
    European = auto()
    American = auto()


@dataclass
class Option:
    pass
