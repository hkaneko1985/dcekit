# -*- coding: utf-8 -*-
"""
Select which FTCP_VAE implementation is used.

`update_regressor_weights` is specified explicitly in the main scripts
(01_main.py, 02_evaluate_all_models.py, 03_compare_conventional_vs_best.py)
and passed to select_ftcp() / build_and_load().

Weights trained with one setting must be evaluated with the same setting,
so keep the value identical in 01, 02 and 03.
"""


def select_ftcp(update_regressor_weights):
    """Return (LossHistoryCallback, FTCP_VAE) of the selected implementation.

    update_regressor_weights : bool
        True  -> module.FTCP_update (regression head is also updated/saved)
        False -> module.FTCP        (regression head is left un-updated)
    """
    if update_regressor_weights:
        from module.FTCP_update import LossHistoryCallback, FTCP_VAE
    else:
        from module.FTCP import LossHistoryCallback, FTCP_VAE
    return LossHistoryCallback, FTCP_VAE
