

from __future__ import annotations
from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Iterable, List, Union


class VectorFeatureLayer(IntEnum):
    """
    Enum for VectorFeatureLayer.
    """

    LANE = 0
    LEFT_BOUNDARY = 1
    RIGHT_BOUNDARY = 2
    STOP_LINE = 3
    CROSSWALK = 4
    ROUTE_LANES = 5

    @classmethod
    def deserialize(cls, layer: str) -> VectorFeatureLayer:
        """Deserialize the type when loading from a string."""
        return VectorFeatureLayer.__members__[layer]



@dataclass
class Point2D:
    """Class to represents 2D points."""

    x: float  # [m] location
    y: float  # [m] location
    __slots__ = "x", "y"

    def __iter__(self) -> Iterable[float]:
        """
        :return: iterator of tuples (x, y)
        """
        return iter((self.x, self.y))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        return np.array([self.x, self.y], dtype=np.float64)

    def __hash__(self) -> int:
        """Hash method"""
        return hash((self.x, self.y))


class TrackedObjectType(Enum):
    """Enum of classification types for TrackedObject."""

    VEHICLE = 0, 'vehicle'
    PEDESTRIAN = 1, 'pedestrian'
    BICYCLE = 2, 'bicycle'
    TRAFFIC_CONE = 3, 'traffic_cone'
    BARRIER = 4, 'barrier'
    CZONE_SIGN = 5, 'czone_sign'
    GENERIC_OBJECT = 6, 'generic_object'
    EGO = 7, 'ego'

    def __int__(self) -> int:
        """
        Convert an element to int
        :return: int
        """
        return self.value  # type: ignore

    def __new__(cls, value: int, name: str) -> TrackedObjectType:
        """
        Create new element
        :param value: its value
        :param name: its name
        """
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name  # type: ignore
        return member

    def __eq__(self, other: object) -> bool:
        """
        Equality checking
        :return: int
        """
        # Cannot check with isisntance, as some code imports this in a different way
        try:
            return self.name == other.name and self.value == other.value  # type: ignore
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash"""
        return hash((self.name, self.value))


class EgoInternalIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
      in the Ego Trajectory Tensors.

    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the ego x position.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the ego y position.
        :return: index
        """
        return 1

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the ego heading.
        :return: index
        """
        return 2

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the ego x velocity.
        :return: index
        """
        return 3

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the ego y velocity.
        :return: index
        """
        return 4

    @staticmethod
    def ax() -> int:
        """
        The dimension corresponding to the ego x acceleration.
        :return: index
        """
        return 5

    @staticmethod
    def ay() -> int:
        """
        The dimension corresponding to the ego y acceleration.
        :return: index
        """
        return 6

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the EgoInternal buffer.
        :return: number of features.
        """
        return 7


class AgentInternalIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
      in the tensors used to compute the final Agent Feature.


    It is intended to be used like an IntEnum, but supported by TorchScript
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def track_token() -> int:
        """
        The dimension corresponding to the track_token for the agent.
        :return: index
        """
        return 0

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the agent.
        :return: index
        """
        return 2

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the agent.
        :return: index
        """
        return 3

    @staticmethod
    def width() -> int:
        """
        The dimension corresponding to the width of the agent.
        :return: index
        """
        return 4

    @staticmethod
    def length() -> int:
        """
        The dimension corresponding to the length of the agent.
        :return: index
        """
        return 5

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x position of the agent.
        :return: index
        """
        return 6

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y position of the agent.
        :return: index
        """
        return 7

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the AgentsInternal buffer.
        :return: number of features.
        """
        return 8
