import enum

class Language(enum.StrEnum):
    python = 'Python'
    r = 'R'
    stata = 'Stata'


class DcdensityResults(enum.StrEnum):
    estimate = 'Density discontinuity (log difference)'
    standard_error = 'Density discontinuity (standard error)'
    z_stat = 'z-statistic'
    p_value = 'p-value'
    bandwidth = 'Bandwidth'
    bin_size = 'Bin size'
    cutoff = 'Cutoff'


class LddtestResults(enum.StrEnum):
    estimate = 'Estimate'
    standard_error = 'Standard error'
    z_stat_equivalence = "z-statistic (equivalence)"
    p_value_equivalence = "p-value (equivalence)"
    confidence_lower_equivalence = 'ECI lower'
    confidence_upper_equivalence = 'ECI upper'
    epsilon_lower = "Epsilon lower"
    epsilon_upper = "Epsilon upper"
    number_observations = "Observations"
    number_observations_effective = "Effective observations"
    bandwidth = 'Bandwidth'