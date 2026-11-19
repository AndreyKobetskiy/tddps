"""Schemas and reference models for the six individual assets."""
from dataclasses import dataclass
from math import sin


def lathe_current(load_pct):
    return 2.5 + 0.12 * load_pct


@dataclass(frozen=True)
class Equipment:
    kind: str
    observers: int
    fields: tuple[str, ...]
    base: tuple[float, ...]
    response: tuple[float, ...]
    noise: tuple[float, ...]

    def state(self, step, machine_id):
        t = step / 10
        phase = t / 75 + 0.7 * machine_id
        demand = 0.55 + 0.22 * sin(phase) + 0.08 * sin(t / 9)
        slow_demand = 0.55 + 0.22 * sin(phase - 0.8)
        result = []
        for index, (field, base, gain) in enumerate(
            zip(self.fields, self.base, self.response, strict=True)
        ):
            if field == "current_a":
                value = lathe_current(result[1])
            else:
                driver = slow_demand if "_temp_" in field else demand
                ripple = 0 if field == "load_pct" else 0.03 * sin(t / (3 + index))
                value = base + gain * (driver + ripple)
            result.append(value)
        return result


EQUIPMENT = (
    Equipment("lathe", 3,
              ("speed_rpm", "load_pct", "motor_temp_c", "current_a", "voltage_v",
               "power_kw", "case_temp_c", "vibration_x_mm_s", "vibration_y_mm_s",
               "vibration_z_mm_s"),
              (950, 0, 39, 0, 397, 1, 30, 0.2, 0.2, 0.1),
              (2200, 100, 22, 0, 6, 7, 15, 2, 1.7, 1.3),
              (3, .25, .12, .04, .25, .025, .10, .012, .012, .012)),
    Equipment("pump", 4,
              ("flow_l_min", "outlet_pressure_bar", "motor_temp_c", "inlet_pressure_bar",
               "impeller_speed_rpm", "power_kw", "fluid_temp_c", "vibration_mm_s"),
              (20, 1.5, 35, 1.8, 600, 0.5, 18, 0.2),
              (130, 4.5, 18, -0.6, 2400, 5, 8, 2),
              (0.4, .025, .10, .015, 4, .025, .08, .015)),
    Equipment("compressor", 3,
              ("discharge_pressure_bar", "airflow_m3_min", "discharge_temp_c",
               "intake_temp_c", "shaft_speed_rpm", "oil_pressure_bar", "power_kw",
               "vibration_mm_s", "dew_point_c"),
              (4, 1, 32, 20, 700, 2, 1, .2, 0),
              (5, 7, 56, 5, 2000, 2, 12, 2, -12),
              (.04, .035, .2, .08, 4, .025, .05, .02, .08)),
    Equipment("furnace", 3,
              ("chamber_temp_c", "heater_power_kw", "exhaust_temp_c", "wall_temp_c",
               "fan_speed_rpm", "airflow_m3_min", "oxygen_pct", "door_surface_temp_c"),
              (220, 2, 90, 55, 500, 1, 20.8, 30),
              (580, 35, 310, 160, 1700, 8, -2, 70),
              (.7, .12, .5, .3, 4, .03, .02, .15)),
    Equipment("conveyor", 3,
              ("belt_speed_m_s", "payload_kg", "drive_temp_c", "belt_tension_n",
               "drive_torque_nm", "drive_power_kw", "roller_vibration_mm_s",
               "tracking_offset_mm", "throughput_kg_min"),
              (0.4, 5, 25, 130, 6, .3, .1, 0, 8),
              (1.6, 95, 28, 470, 44, 4, 1.8, 3, 280),
              (.01, .3, .12, 2, .2, .02, .015, .025, 1)),
    Equipment("chiller", 3,
              ("supply_temp_c", "return_temp_c", "coolant_flow_l_min",
               "evaporator_pressure_bar", "condenser_pressure_bar", "fan_speed_rpm",
               "cooling_capacity_kw", "power_kw"),
              (8, 9, 15, 2.5, 6, 400, 2, .5),
              (-4, 8, 90, 2, 4, 1400, 25, 8),
              (.05, .08, .3, .02, .04, 4, .1, .04)),
)
