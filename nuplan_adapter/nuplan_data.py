

from __future__ import annotations
from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Iterable, List, Union, Dict, Any
import numpy as np


class TrafficLightStatusType(IntEnum):
    """
    Enum for TrafficLightStatusType.
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = 3

    def serialize(self) -> str:
        """Serialize the type when saving."""
        return self.name

    @classmethod
    def deserialize(cls, key: str) -> TrafficLightStatusType:
        """Deserialize the type when loading from a string."""
        return TrafficLightStatusType.__members__[key]


@dataclass
class TrafficLightStatusData:
    """Traffic light status."""

    status: TrafficLightStatusType  # Status: green, red
    lane_connector_id: int  # lane connector id, where this traffic light belongs to
    timestamp: int  # Timestamp

    def serialize(self) -> Dict[str, Any]:
        """Serialize traffic light status."""
        return {
            'status': self.status.serialize(),
            'lane_connector_id': self.lane_connector_id,
            'timestamp': self.timestamp,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrafficLightStatusData:
        """Deserialize a dict of data to this class."""
        return TrafficLightStatusData(
            status=TrafficLightStatusType.deserialize(data['status']),
            lane_connector_id=data['lane_connector_id'],
            timestamp=data['timestamp'],
        )


class SemanticMapLayer(IntEnum):
    """
    Enum for SemanticMapLayers.
    """

    LANE = 0
    INTERSECTION = 1
    STOP_LINE = 2
    TURN_STOP = 3
    CROSSWALK = 4
    DRIVABLE_AREA = 5
    YIELD = 6
    TRAFFIC_LIGHT = 7
    STOP_SIGN = 8
    EXTENDED_PUDO = 9
    SPEED_BUMP = 10
    LANE_CONNECTOR = 11
    BASELINE_PATHS = 12
    BOUNDARIES = 13
    WALKWAYS = 14
    CARPARK_AREA = 15
    PUDO = 16
    ROADBLOCK = 17
    ROADBLOCK_CONNECTOR = 18

    @classmethod
    def deserialize(cls, layer: str) -> SemanticMapLayer:
        """Deserialize the type when loading from a string."""
        return SemanticMapLayer.__members__[layer]




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



@dataclass
class MapObjectPolylines:
    """
    Collection of map object polylines, each represented as a list of x, y coords
    [num_elements, num_points_in_element (variable size), 2].
    """

    polylines: List[List[Point2D]]

    def to_vector(self) -> List[List[List[float]]]:
        """
        Returns data in vectorized form
        :return: vectorized coords of map object polylines as [num_elements, num_points_in_element (variable size), 2].
        """
        return [[[coord.x, coord.y] for coord in polygon] for polygon in self.polylines]
    
    
@dataclass
class LaneSegmentTrafficLightData:
    """
    Traffic light data represented as one-hot encoding per segment [num_lane_segment, 4].
    The one-hot encoding: green [1, 0, 0, 0], yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1].
    """

    traffic_lights: List[Tuple[int, int, int, int]]

    _one_hot_encoding = {
        TrafficLightStatusType.GREEN: (1, 0, 0, 0),
        TrafficLightStatusType.YELLOW: (0, 1, 0, 0),
        TrafficLightStatusType.RED: (0, 0, 1, 0),
        TrafficLightStatusType.UNKNOWN: (0, 0, 0, 1),
    }
    _encoding_dim: int = 4

    def to_vector(self) -> List[List[float]]:
        """
        Returns data in vectorized form.
        :return: vectorized traffic light data per segment as [num_lane_segment, 4].
        """
        return [list(data) for data in self.traffic_lights]

    @classmethod
    def encode(cls, traffic_light_type: TrafficLightStatusType) -> Tuple[int, int, int, int]:
        """
        One-hot encoding of TrafficLightStatusType: green [1, 0, 0, 0], yellow [0, 1, 0, 0], red [0, 0, 1, 0],
            unknown [0, 0, 0, 1].
        """
        return cls._one_hot_encoding[traffic_light_type]

    @classmethod
    def encoding_dim(cls) -> int:
        """
        Dimensionality of associated data encoding.
        :return: encoding dimensionality.
        """
        return cls._encoding_dim

@dataclass
class LaneSegmentLaneIDs:
    """
    IDs of lane/lane connectors that lane segment at specified index belong to.
    """

    lane_ids: List[str]
    
class AgentFeatureIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
        in the final output agents feature.

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
        The dimension corresponding to the x coordinate of the agent.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y coordinate of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the agent.
        :return: index
        """
        return 2

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the agent.
        :return: index
        """
        return 3

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the agent.
        :return: index
        """
        return 4

    @staticmethod
    def yaw_rate() -> int:
        """
        The dimension corresponding to the yaw rate of the agent.
        :return: index
        """
        return 5

    @staticmethod
    def length() -> int:
        """
        The dimension corresponding to the length of the agent.
        :return: index
        """
        return 6

    @staticmethod
    def width() -> int:
        """
        The dimension corresponding to the width of the agent.
        :return: index
        """
        return 7

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the AgentsFeature.
        :return: number of features.
        """
        return 8


class StateVector2D:
    """Representation of vector in 2d."""

    __slots__ = "_x", "_y", "_array"

    def __init__(self, x: float, y: float):
        """
        Create StateVector2D object
        :param x: float direction
        :param y: float direction
        """
        self._x = x  # x-axis in the vector.
        self._y = y  # y-axis in the vector.

        self._array: npt.NDArray[np.float64] = np.array([self.x, self.y], dtype=np.float64)

    def __repr__(self) -> str:
        """
        :return: string containing representation of this class
        """
        return f'x: {self.x}, y: {self.y}'

    def __eq__(self, other: object) -> bool:
        """
        Compare other object with this class
        :param other: object
        :return: true if other state vector is the same as self
        """
        if not isinstance(other, StateVector2D):
            return NotImplemented
        return bool(np.array_equal(self.array, other.array))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """
        Convert vector to array
        :return: array containing [x, y]
        """
        return self._array

    @array.setter
    def array(self, other: npt.NDArray[np.float64]) -> None:
        """Custom setter so that the object is not corrupted."""
        self._array = other
        self._x = other[0]
        self._y = other[1]

    @property
    def x(self) -> float:
        """
        :return: x float state
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        """Custom setter so that the object is not corrupted."""
        self._x = x
        self._array[0] = x

    @property
    def y(self) -> float:
        """
        :return: y float state
        """
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        """Custom setter so that the object is not corrupted."""
        self._y = y
        self._array[1] = y

    def magnitude(self) -> float:
        """
        :return: magnitude of vector
        """
        return float(np.hypot(self.x, self.y))


class TimeDuration:
    """Class representing a time delta, with a microsecond resolution."""

    __slots__ = "_time_us"

    def __init__(self, *, time_us: int, _direct: bool = True) -> None:
        """Constructor, should not be called directly. Raises if the keyword parameter _direct is not set to false."""
        if _direct:
            raise RuntimeError("Don't initialize this class directly, use one of the constructors instead!")

        self._time_us = time_us

    @classmethod
    def from_us(cls, t_us: int) -> TimeDuration:
        """
        Constructs a TimeDuration from a value in microseconds.
        :param t_us: Time in microseconds.
        :return: TimeDuration.
        """
        assert isinstance(t_us, int), "Microseconds must be an integer!"
        return cls(time_us=t_us, _direct=False)

    @classmethod
    def from_ms(cls, t_ms: float) -> TimeDuration:
        """
        Constructs a TimeDuration from a value in milliseconds.
        :param t_ms: Time in milliseconds.
        :return: TimeDuration.
        """
        return cls(time_us=int(t_ms * int(1e3)), _direct=False)

    @classmethod
    def from_s(cls, t_s: float) -> TimeDuration:
        """
        Constructs a TimeDuration from a value in seconds.
        :param t_s: Time in seconds.
        :return: TimeDuration.
        """
        return cls(time_us=int(t_s * int(1e6)), _direct=False)

    @property
    def time_us(self) -> int:
        """
        :return: TimeDuration in microseconds.
        """
        return self._time_us

    @property
    def time_ms(self) -> float:
        """
        :return: TimeDuration in milliseconds.
        """
        return self._time_us / 1e3

    @property
    def time_s(self) -> float:
        """
        :return: TimeDuration in seconds.
        """
        return self._time_us / 1e6

    def __add__(self, other: object) -> TimeDuration:
        """
        Adds a time duration to a time duration.
        :param other: time duration.
        :return: self + other if other is a TimeDuration.
        """
        if isinstance(other, TimeDuration):
            return TimeDuration.from_us(self.time_us + other.time_us)
        return NotImplemented

    def __sub__(self, other: object) -> TimeDuration:
        """
        Subtract a time duration from a time duration.
        :param other: time duration.
        :return: self - other if other is a TimeDuration.
        """
        if isinstance(other, TimeDuration):
            return TimeDuration.from_us(self.time_us - other.time_us)
        return NotImplemented

    def __mul__(self, other: object) -> TimeDuration:
        """
        Multiply a time duration by a scalar value.
        :param other: value to multiply.
        :return: self * other if other is a scalar.
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s * other)
        return NotImplemented

    def __rmul__(self, other: object) -> TimeDuration:
        """
        Multiply a time duration by a scalar value.
        :param other: value to multiply.
        :return: self * other if other is a scalar.
        """
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented

    def __truediv__(self, other: object) -> TimeDuration:
        """
        Divides a time duration by a scalar value.
        :param other: value to divide for.
        :return: self / other if other is a scalar.
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s / other)
        return NotImplemented

    def __floordiv__(self, other: object) -> TimeDuration:
        """
        Floor divides a time duration by a scalar value.
        :param other: value to divide for.
        :return: self // other if other is a scalar.
        """
        if isinstance(other, (int, float)):
            return TimeDuration.from_s(self.time_s // other)
        return NotImplemented

    def __gt__(self, other: TimeDuration) -> bool:
        """
        Self is greater than other.
        :param other: TimeDuration.
        :return: True if self > other, False otherwise.
        """
        if isinstance(other, TimeDuration):
            return self.time_us > other.time_us
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        """
        Self is greater or equal than other.
        :param other: TimeDuration.
        :return: True if self >= other, False otherwise.
        """
        if isinstance(other, TimeDuration):
            return self.time_us >= other.time_us
        return NotImplemented

    def __lt__(self, other: TimeDuration) -> bool:
        """
        Self is less than other.
        :param other: TimeDuration.
        :return: True if self < other, False otherwise.
        """
        if isinstance(other, TimeDuration):
            return self.time_us < other.time_us
        return NotImplemented

    def __le__(self, other: TimeDuration) -> bool:
        """
        Self is less or equal than other.
        :param other: TimeDuration.
        :return: True if self <= other, False otherwise.
        """
        if isinstance(other, TimeDuration):
            return self.time_us <= other.time_us
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """
        Self is equal to other.
        :param other: TimeDuration.
        :return: True if self == other, False otherwise.
        """
        if not isinstance(other, TimeDuration):
            return NotImplemented

        return self.time_us == other.time_us

    def __hash__(self) -> int:
        """
        :return: hash for this object.
        """
        return hash(self.time_us)

    def __repr__(self) -> str:
        """
        :return: String representation.
        """
        return "TimeDuration({}s)".format(self.time_s)


@dataclass
class TimePoint:
    """
    Time instance in a time series.
    """

    time_us: int  # [micro seconds] time since epoch in micro seconds
    __slots__ = "time_us"

    def __post_init__(self) -> None:
        """
        Validate class after creation.
        """
        assert self.time_us >= 0, "Time point has to be positive!"

    @property
    def time_s(self) -> float:
        """
        :return [s] time in seconds.
        """
        return self.time_us * 1e-6

    def __add__(self, other: object) -> TimePoint:
        """
        Adds a TimeDuration to generate a new TimePoint.
        :param other: time point.
        :return: self + other.
        """
        if isinstance(other, (TimeDuration, TimePoint)):
            return TimePoint(self.time_us + other.time_us)
        return NotImplemented

    def __radd__(self, other: object) -> TimePoint:
        """
        :param other: Right addition target.
        :return: Addition with other if other is a TimeDuration.
        """
        if isinstance(other, TimeDuration):
            return self.__add__(other)
        return NotImplemented

    def __sub__(self, other: object) -> TimePoint:
        """
        Subtract a time duration from a time point.
        :param other: time duration.
        :return: self - other if other is a TimeDuration.
        """
        if isinstance(other, (TimeDuration, TimePoint)):
            return TimePoint(self.time_us - other.time_us)
        return NotImplemented

    def __gt__(self, other: TimePoint) -> bool:
        """
        Self is greater than other.
        :param other: time point.
        :return: True if self > other, False otherwise.
        """
        if isinstance(other, TimePoint):
            return self.time_us > other.time_us
        return NotImplemented

    def __ge__(self, other: TimePoint) -> bool:
        """
        Self is greater or equal than other.
        :param other: time point.
        :return: True if self >= other, False otherwise.
        """
        if isinstance(other, TimePoint):
            return self.time_us >= other.time_us
        return NotImplemented

    def __lt__(self, other: TimePoint) -> bool:
        """
        Self is less than other.
        :param other: time point.
        :return: True if self < other, False otherwise.
        """
        if isinstance(other, TimePoint):
            return self.time_us < other.time_us
        return NotImplemented

    def __le__(self, other: TimePoint) -> bool:
        """
        Self is less or equal than other.
        :param other: time point.
        :return: True if self <= other, False otherwise.
        """
        if isinstance(other, TimePoint):
            return self.time_us <= other.time_us
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        """
        Self is equal to other
        :param other: time point
        :return: True if self == other, False otherwise
        """
        if not isinstance(other, TimePoint):
            return NotImplemented

        return self.time_us == other.time_us

    def __hash__(self) -> int:
        """
        :return: hash for this object
        """
        return hash(self.time_us)

    def diff(self, time_point: TimePoint) -> TimeDuration:
        """
        Computes the TimeDuration between self and another TimePoint.
        :param time_point: The other time point.
        :return: The TimeDuration between the two TimePoints.
        """
        return TimeDuration.from_us(int(self.time_us - time_point.time_us))

@dataclass
class VectorFeatureLayerMapping:
    """
    Dataclass for associating VectorFeatureLayers with SemanticMapLayers for extracting map object polygons.
    """

    _semantic_map_layer_mapping = {
        VectorFeatureLayer.STOP_LINE: SemanticMapLayer.STOP_LINE,
        VectorFeatureLayer.CROSSWALK: SemanticMapLayer.CROSSWALK,
    }

    @classmethod
    def available_polygon_layers(cls) -> List[VectorFeatureLayer]:
        """
        List of VectorFeatureLayer for which mapping is supported.
        :return List of available layers.
        """
        return list(cls._semantic_map_layer_mapping.keys())

    @classmethod
    def semantic_map_layer(cls, feature_layer: VectorFeatureLayer) -> SemanticMapLayer:
        """
        Returns associated SemanticMapLayer for feature extraction, if exists.
        :param feature_layer: specified VectorFeatureLayer to look up.
        :return associated SemanticMapLayer.
        """
        return cls._semantic_map_layer_mapping[feature_layer]

