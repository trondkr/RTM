import pandas as pd
import gcsfs
import numpy as np
import logging

class Config_robustness():

    def __init__(self):
        """
        Class to use for calculating he robustness between climate model predictions. Where
        and when do light projections agree across models.
        """
        logging.info("[CMIP6_robustness] Initializing the robustness calculations")