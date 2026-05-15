from dataclasses import dataclass

@dataclass(frozen=True)
class HueParams:
    hue_value: float
    hue_width: float
    saturation_minimum: float = 0.05
    intensity_upper_limit: float = 0.95
    intensity_weak_threshold: float = 0.75
    intensity_strong_threshold: float = 0.45
    intensity_lower_limit: float = 0.05