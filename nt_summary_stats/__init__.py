"""
NT Summary Stats - Fast neutrino telescope summary statistics computation.

This package provides efficient computation of the 9 traditional summary statistics
for neutrino telescope sensors (optical modules), as described in the IceCube paper
(https://arxiv.org/abs/2101.11589).

The 9 summary statistics are:
1. Total DOM charge
2. Charge within 100ns of first pulse  
3. Charge within 500ns of first pulse
4. Time of first pulse
5. Time of last pulse
6. Time at which 20% of charge is collected
7. Time at which 50% of charge is collected
8. Charge weighted mean time
9. Charge weighted standard deviation time
"""

from .core import compute_summary_stats
from .prometheus import process_prometheus_event, process_sensor_data

__version__ = "0.1.3"
__all__ = ["compute_summary_stats", "process_prometheus_event", "process_sensor_data"]