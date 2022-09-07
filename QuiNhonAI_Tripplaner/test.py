from http.client import UNSUPPORTED_MEDIA_TYPE
import pandas as pd
from ast import literal_eval

budget_file = "./data/Sample/budget.csv"
location_file = "./data/Sample/location.csv"
roadtrip_file = "./data/Sample/roadtrip.csv"
transport_file = "./data/Sample/transport.csv"
usertrip_file = "./data/Sample/usertrip.csv"


usertrip = pd.read_csv(usertrip_file)
usertrip.arrivalTime = usertrip.arrivalTime.apply(literal_eval)
usertrip.duration = usertrip.duration.apply(literal_eval)

budget = pd.read_csv(budget_file).to_dict('records')[0]
location = pd.read_csv(location_file).to_dict('records')

location = {x['locationID']: x for x in location}

# location2id = {location[i]['locationID']: i for i in location}

roadtrip = pd.read_csv(roadtrip_file).to_dict('records')
roadtrip = {x['Unnamed: 0']: x for x in roadtrip}


transport = pd.read_csv(transport_file).to_dict('records')
transport = {x['transportationID']: x for x in transport}
print(transport)
