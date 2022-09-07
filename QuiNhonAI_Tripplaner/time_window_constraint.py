from math import factorial
import pandas as pd
from ast import literal_eval
import itertools
import numpy as np
import pandas as pd

class Solver:
    """
    This simple solver will just go through all the locations in order
    And try to fit them into the schedule.
    """
    def __init__(self, usertrip_csv, location_csv, roadtrip_csv, transport_csv, budget_csv):
        self.usertrip_csv = usertrip_csv
        self.location_csv = location_csv
        self.roadtrip_csv = roadtrip_csv
        self.transport_csv = transport_csv
        self.budget_csv = budget_csv 
    
        self.location_arr = usertrip_csv['locationID'].unique() 
        self.person_count = len(usertrip_csv['userName'].unique())