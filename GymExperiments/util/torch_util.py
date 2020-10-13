def get_mode_normal(distribution):
    return distribution.mean

def get_mode_beta(distribution):
    conc0 = distribution.concentration0
    conc1 = distribution.concentration1
    mode = (conc0 - 1) / (conc0 + conc1 - 2)
    return mode