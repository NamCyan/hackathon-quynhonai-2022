#!/usr/bin/python
#
# Copyright QAI Fsoft, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    def suitable_fitness(self, arrival_time_list, current_nextarrival_time):
        """
        Check the suitable fitness is in the arrival_time_list based on current_nextarrival_time
        """
        arrival_time_list = arrival_time_list.tolist()
        arrival_time_list.sort()
        lower_arr = []
        upper_arr = []
        for arrival_time in arrival_time_list:
            arrival_time = np.fromstring(arrival_time[1:-1], dtype=float, sep=',')
            lower_bound = arrival_time[0]
            upper_bound = arrival_time[1]
            if lower_bound > current_nextarrival_time:
                lower_arr.append(lower_bound)
            if upper_bound > current_nextarrival_time:
                upper_arr.append(upper_bound)

        # Get min of lower_arr
        if len(lower_arr) == 0:
            if len(upper_arr) == 0:
                return current_nextarrival_time + 0.1       
        return min(lower_arr)

    def best_fitness(self,array_list):
        """
        Try to find the best fitness in the array_list
        """
        array_list = array_list.tolist()
        array_list.sort()
        # Get the average number of array_list
        best_count = 0
        best_fitness = -1
        for arrival_time in array_list:
            # Convert to numpy array
            arrival_time = np.fromstring(arrival_time[1:-1], dtype=float, sep=',')
            current_count = 0
            best_number = -1
            for i in np.arange(arrival_time[0],arrival_time[1],0.1):
                # Check i is in other range of array_list
                count = 0
                for j in array_list:
                    j = np.fromstring(j[1:-1], dtype=float, sep=',')
                    if j[0] != arrival_time[0] and j[1] != arrival_time[1]:
                        if j[0] <= i <= j[1]:
                            count += 1
                if count > current_count:
                    current_count = count
                    best_number = i

            if current_count > best_count:
                best_count = current_count
                best_fitness = best_number             
        if best_fitness == -1:
            # print("No best arrival time found!")
            best_possible_arrival_time = array_list[0]
            best_possible_arrival_time = best_possible_arrival_time[1:-1]
            best_possible_arrival_time = np.fromstring(best_possible_arrival_time, dtype=float, sep=',')
            best_fitness = best_possible_arrival_time[0]

        return best_fitness

    def get_distance(self, start_location, end_location):
        """
        Get the distance between start_location and end_location
        """
        start_location = int(start_location)
        end_location = int(end_location)
        distance_first_column = self.roadtrip_csv.iloc[:,0]
        index = distance_first_column.index[distance_first_column == start_location]
        value = self.roadtrip_csv[str(end_location)].iloc[index].values[0]   

        return value

    def solve(self, location_factorial, person_count):
        """
        Brute force to find the best solution
        """ 
        print("Start to solve the problem!")

        days = self.budget_csv['days'].iloc[0]

        for location_list in location_factorial:
            print(location_list)
            # Convert to list
            location_list = list(location_list)

            home_location = location_list[0]
            # Remove home_location from location_list
            location_list.remove(home_location)

            total_budget = 0
            current_nextarrival_time = -1
            day_for_location = {}
            day_th = 1
            planning = {}
            temp = -1

            # Add home to the last location, to create a cycle
            location_list.append(home_location)

            # The final solution is not nessessary to contain all location in location_list
            # So, you can optimize the solution
            for index,location_id in enumerate(location_list):
                usertrip_filter = self.usertrip_csv[self.usertrip_csv['locationID'] == location_id]
                arrival_time_list = usertrip_filter['arrivalTime']
                duration_time_list = usertrip_filter['duration']

                # Find the best number fit one of the arrival_time
                bestarrival_time = self.best_fitness(arrival_time_list)
                bestduration_time = self.best_fitness(duration_time_list) 

                print(bestarrival_time, bestduration_time)
                if location_id != temp:
                    # Check bestarrival_time of location_id
                    if bestarrival_time < current_nextarrival_time:
                        # Get maximum values of day_for_location
                        max_day = max(day_for_location.values()) if day_for_location else 0                      
                        # We don't have chance
                        if max_day != 0 and max_day == days:
                            # We will choose the best suitable arrival time in this case
                            # Nearest to current_nextarrival_time
                            bestarrival_time = self.suitable_fitness(arrival_time_list, current_nextarrival_time)
                        else:
                            # We can move this location to the next day
                            # Before go to next day, we will add home location to planning
                            # Add home location to index of location_list
                            # Remember that, when we move this location to the next day
                            # We will comeback to home location
                            if location_id != home_location:
                                temp = location_id
                                location_id = home_location
                                location_list[index] = home_location
                                location_list.insert(index+1, temp)
                                day_th += 1
                # Move location_id to next day
                day_for_location[location_id] = day_th                
                if index == 0:
                    distance = self.get_distance(home_location, location_id)
                    final_name = str(home_location) + "->" + str(location_id)
                else:
                    distance = self.get_distance(location_list[index-1],location_id)
                    final_name = str(location_list[index-1]) + "->" + str(location_id)      

                # Check budget of location
                # In this solver, we only use transportID = 1, that means we only use car
                # So if you want to optimize this solver 
                # You can add any conditions related to transportID
                # Get cost of location       
                location_filter = self.location_csv[self.location_csv['locationID'] == location_id]        
                minprice_location = location_filter['minPrice'].iloc[0] * person_count

                # In this solver, we don't optimize arrival_time follow by openTime and closeTime of location
                # So if you want to optimize this solver
                # You can add any conditions related to openTime and closeTime of location
                transport_id = 1
                transport_filter = self.transport_csv[self.transport_csv['transportationID'] == transport_id]
                cost_transport = transport_filter['cost'].values[0]
                # Because distance is in m, we need to convert to km to calculate cost

                distance = self.get_distance(location_list[index-1], location_id) / 1000
                cost_transport = distance * cost_transport
                cost_transport = cost_transport * person_count
                speed_transport = transport_filter['speed'].iloc[0] * 1000 # Convert km to m

                # Calculate time for location               
                if index == len(location_list) - 1:
                    next_distance = 0.0
                else: 
                    next_distance = self.get_distance(location_id, location_list[index+1]) / 1000 # Convert m to km

                time_travel = next_distance / speed_transport
                best_nextarrival_time = bestarrival_time + bestduration_time + time_travel
                best_spend_money = minprice_location 
                total_budget += best_spend_money + cost_transport

                # Get maximum values of day_for_location
                max_day = max(day_for_location.values()) if day_for_location else 0
                result = [day_th,bestarrival_time, bestduration_time, best_spend_money, transport_id]    
                if best_nextarrival_time > 24 and max_day == days:
                    # Can't choose any suitable location in this case
                    # best_nextarrival_time = 24.00
                    # Remove the rest of location in location_list
                    # location_list = location_list[:index+1]
                    bestarrival_time = current_nextarrival_time + 0.01

                    if bestarrival_time > 24:
                        bestarrival_time = 23.99
                        bestduration_time = 0.5              
                    planning[final_name] = result                     
                    break

                current_nextarrival_time = best_nextarrival_time
                planning[final_name] = result   

            # Add final row
            # Please remember that, that last row is not important
            # Because the evaluation function will not use it
            # But we still check its logic here to make sure that the solver is correct

            last_element = list(planning.values())[-1]
            day_th, bestarrival_time, bestduration_time, best_spend_money, transport_id = last_element
            planning[str(location_list[len(location_list) - 1]) + '->' + str(0)] = [day_th, 6.0, 0.5, 100, transport_id]
            submission = self.save_solution(planning)

            return submission

    def save_solution(self,planning):
        """
        Save solution to csv file
        """
        submission = pd.DataFrame(columns=['noDay', 'locationID', 'arrivalTime', 'duration', 'spendMoney', 'transportID'])
        for key, value in planning.items():
            no_day, arrival_time, duration, spend_money, transport_id = value
            location_id = key.split('->')[0]
            submission = submission.append({'noDay': int(no_day), 'locationID': int(location_id), 'arrivalTime': round(float(arrival_time),2), 'duration': round(float(duration),2), 'spendMoney': round(float(spend_money),2), 'transportID': int(transport_id)}, ignore_index=True)

        return submission

def main():
    """
    Main function
    """
    import warnings
    warnings.filterwarnings("ignore")
    print("Start to solve the problem!")
    path = '../data/Sample/'
    outputpath = './sample_submission/'

    # Read data
    usertrip_csv = pd.read_csv(path + 'usertrip.csv')
    location_csv = pd.read_csv(path + 'location.csv')
    roadtrip_csv = pd.read_csv(path +  'roadtrip.csv')
    transport_csv = pd.read_csv(path + 'transport.csv')
    budget_csv = pd.read_csv(path + 'budget.csv')

    # Combine all to one node
    location_arr = usertrip_csv['locationID'].unique() 
    person_count = len(usertrip_csv['userName'].unique())

    location_factorial = list(itertools.permutations(location_arr))
    # print(location_factorial)

    print("Solving...")

    mysolver = Solver(usertrip_csv, location_csv, roadtrip_csv, transport_csv, budget_csv)

    submission_csv = mysolver.solve([location_factorial[0]],person_count)

    # Save to submission
    submission_csv.to_csv(outputpath + 'submission.csv', index=False)

if __name__ == "__main__":
    main()