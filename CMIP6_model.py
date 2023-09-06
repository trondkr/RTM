import sys
import logging
# Local files and utility functions
sys.path.append("./subroutines/")


class CMIP6_MODEL():
    def __init__(self, name):
        self.name = name
        self.ocean_vars = {}
        self.ds_sets = {}
        self.member_ids = []
        self.current_time = None
        self.current_member_id = None
        self.experiment_id = None

    def description(self):
        logging.info("[CMIP6_model] --------------")
        logging.info("[CMIP6_model] {} ".format(self.name))
        for ds in self.ds_sets.keys():
            logging.info("[CMIP6_model] Model dataset: {} ".format(ds))
        logging.info("[CMIP6_model] members: {}".format(self.member_ids))
        logging.info("[CMIP6_model] variables: {}".format(self.ocean_vars))
        logging.info("[CMIP6_model] --------------")
