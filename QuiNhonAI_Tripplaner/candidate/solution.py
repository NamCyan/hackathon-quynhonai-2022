import pandas as pd
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

        self.get_info()

        self.location_arr = list(usertrip_csv['locationID'].unique())
        self.full_locations = self.location_arr.copy()
        self.person_count = len(usertrip_csv['userName'].unique())

        self.user_preference_location= {}
        self.user_arrival_duration_time = {}
        for location_id in self.location_arr:
            usertrip_filter = self.usertrip_csv[self.usertrip_csv['locationID'] == location_id].sort_values(by=['arrivalTime'])
            # preference
            
            self.user_preference_location[location_id] = list(usertrip_filter.preference)
            
            # duration
            arrival_time_list = list(usertrip_filter['arrivalTime'])
            duration_time_list = list(usertrip_filter['duration'])
            assert len(arrival_time_list) == len(duration_time_list)

            self.user_arrival_duration_time[location_id] = {}
            self.user_arrival_duration_time[location_id]['arrivalTime'] = []
            self.user_arrival_duration_time[location_id]['duration'] = []
            
            for i, arrival_time in enumerate(arrival_time_list):
                arrival_time = np.fromstring(arrival_time[1:-1], dtype=float, sep=',')
                duration_time = np.fromstring(duration_time_list[i][1:-1], dtype=float, sep=',')

                self.user_arrival_duration_time[location_id]['arrivalTime'].append(list(arrival_time))
                self.user_arrival_duration_time[location_id]['duration'].append(list(duration_time))

        # Get best time for arrival and duration
        for location_id in self.user_arrival_duration_time:
            self.user_arrival_duration_time[location_id]['best_arrivalTime'] = self.best_fitness(self.user_arrival_duration_time[location_id]['arrivalTime'])
            self.user_arrival_duration_time[location_id]['best_duration'] = self.best_fitness(self.user_arrival_duration_time[location_id]['duration']) 
            self.user_arrival_duration_time[location_id]['best_score'] = self.evaluate_single_location(location_id, self.user_arrival_duration_time[location_id]['best_arrivalTime'], self.user_arrival_duration_time[location_id]['best_duration'], self.location_csv[location_id]['minPrice']*self.person_count)

        # print(self.user_arrival_duration_time)
        # Get max days
        self.max_days = self.budget_csv['days']
        self.max_hours = self.max_days * 12

        self.location_arr = set(self.location_arr) - set(self.extract_bad_locations())
        # if not evaluate home location should we keep all for home location choices
    
    def get_info(self):
        self.budget_csv = self.budget_csv.to_dict('records')[0]
        location = self.location_csv.to_dict('records')

        self.location_csv = {x['locationID']: x for x in location}

        roadtrip = self.roadtrip_csv.to_dict('records')
        self.roadtrip_csv = {str(x['Unnamed: 0']): x for x in roadtrip}

        transport = self.transport_csv.to_dict('records')
        self.transport_csv = {x['transportationID']: x for x in transport}

    def extract_bad_locations(self):
        """Remove location that has best score is negative"""
        bad_locations = []
        for location_id in self.location_arr:
            if self.user_arrival_duration_time[location_id]['best_score'] < 0:
                bad_locations.append(location_id)
        return bad_locations
    
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
        # array_list = array_list.tolist()
        # array_list.sort()
        # Get the average number of array_list
        best_score = -1e5
        best_fitness = -1
        for arrival_time in array_list:
            # Convert to numpy array
            # arrival_time = np.fromstring(arrival_time[1:-1], dtype=float, sep=',')
            current_score = -1e5
            best_number = -1

            for i in np.arange(arrival_time[0], arrival_time[1] + 0.01, 0.01):
                # Check i is in other range of array_list
                score = 0
                for j in array_list:
                    # j = np.fromstring(j[1:-1], dtype=float, sep=',')
                    
                    if j[0] <= i <= j[1]:
                        score += 5
                    elif j[0] > i:
                        score -= 25 * (j[0] - i)
                    elif i > j[1]:
                        score -= 25 * (i - j[1])


                if score > current_score:
                    current_score = score
                    best_number = i

            if current_score > best_score:
                best_score = current_score
                best_fitness = best_number   

    
        # if best_fitness == -1:
        #     # print("No best arrival time found!")
        #     best_possible_arrival_time = array_list[0]
        #     best_possible_arrival_time = best_possible_arrival_time[1:-1]
        #     best_possible_arrival_time = np.fromstring(best_possible_arrival_time, dtype=float, sep=',')
        #     best_fitness = best_possible_arrival_time[0]

        return best_fitness

    def get_distance(self, start_location, end_location):
        """
        Get the distance between start_location and end_location
        """

        return self.roadtrip_csv[str(start_location)][str(end_location)] 

    def search_next_location(self, remain_locations, pre_location, pre_accumulated_time):
        best_score = -1e5
        current_best = None
        score = 0
        for next_location in remain_locations:
            for transport_id in self.transport_csv:
                distance = self.get_distance(pre_location, next_location) / 1000
                arrivalTime = pre_accumulated_time + distance / self.transport_csv[transport_id]['speed']
                duration = self.user_arrival_duration_time[next_location]['best_duration']

                # transport_cost = distance * self.transport_csv[transport_id]['cost'] * self.person_count

                # if self.location_csv[next_location]['minPrice'] == 0:
                #     all_cost = transport_cost
                # else:
                #     all_cost = self.location_csv[next_location]['minPrice'] * self.person_count + transport_cost + 1e-6

                if self.location_csv[next_location]['minPrice'] == 0:
                    all_cost = 0.0
                else:
                    all_cost = self.location_csv[next_location]['minPrice'] * self.person_count + 1e-6

                if arrivalTime > 24:
                    arrivalTime -= 24 * (arrivalTime // 24)

                score = self.evaluate_single_location(next_location, arrivalTime, duration, all_cost)
                if score > best_score:
                    best_score = score
                    current_best = {'location': next_location, 'arrivalTime': arrivalTime, 'duration': duration, 'cost': all_cost, 'transport': transport_id, 'score': score}

        remain_locations.remove(current_best['location'])

        return current_best, remain_locations

    def fix_total_cost(self, planning):
        minCost, maxCost = self.budget_csv['minWallet'], self.budget_csv['maxWallet']
        totalCost = 0
        for i, (key, value) in enumerate(planning.items()):
            if i == (len(planning) - 1):
                continue
            no_day, arrival_time, duration, spend_money, transport_id = value
            pre_location = int(key.split("->")[0])
            current_location = int(key.split("->")[1])

            transport_cost = self.transport_csv[transport_id]['cost'] * (self.get_distance(pre_location, current_location) / 1000) * self.person_count

            totalCost += float(spend_money) + transport_cost
        
        if totalCost < minCost:
            for loc2loc in list(planning.keys())[:-1]:
                location_id = int(loc2loc.split("->")[1])
                if self.location_csv[location_id]['maxPrice'] == 0:
                    continue

                old_spend_money = planning[loc2loc][3]

                changes = (self.location_csv[location_id]['maxPrice'] - self.location_csv[location_id]['minPrice'] - 1e-6) * self.person_count
                new_spend_money = old_spend_money + changes
                tmp_total_cost = totalCost + changes
                
                if tmp_total_cost > maxCost:
                    changes_need = maxCost - totalCost - 1e-6 
                    new_spend_money = old_spend_money + changes_need
                    tmp_total_cost = totalCost + changes_need
                    
                planning[loc2loc][3] = new_spend_money
                totalCost = tmp_total_cost

                if totalCost >= minCost:
                    break

        return planning

    def solve_single(self, home_location, unvisited_locations):

        location_list = unvisited_locations.copy()

        best_positive_score = -1e5
        best_planning = None

        for first_visited_location in location_list:
            planning = {}
            # Visit the first place

            arrivalTime = self.user_arrival_duration_time[first_visited_location]['best_arrivalTime'] 
            duration = self.user_arrival_duration_time[first_visited_location]['best_duration']

            transport_id = 1 # does not matter in the first location

            # transport_cost = self.transport_csv[transport_id]['cost'] * (self.get_distance(home_location, first_visited_location) / 1000) * self.person_count
            
            # if self.location_csv[first_visited_location]['minPrice'] == 0:
            #     all_cost = transport_cost
            # else:
            #     all_cost = (self.location_csv[first_visited_location]['minPrice'])* self.person_count + transport_cost + 1e-6
            
            if self.location_csv[first_visited_location]['minPrice'] == 0:
                all_cost = 0
            else:
                all_cost = self.location_csv[first_visited_location]['minPrice']* self.person_count + 1e-6

            score = self.evaluate_single_location(first_visited_location, arrivalTime, duration, all_cost)
            
            
            result = {'arrivalTime': arrivalTime, 'duration': duration, 'cost': all_cost, 'transport': transport_id, 'score': score}

            result_name = "{}->{}".format(home_location, first_visited_location)
            planning[result_name] = result

            remain_locations = location_list.copy()
            remain_locations.remove(first_visited_location)

            pre_location = first_visited_location
            pre_accumulated_time = arrivalTime + duration            
            # Searching route

            while len(remain_locations) > 0:
                next_visited_location, remain_locations = self.search_next_location(remain_locations, pre_location, pre_accumulated_time)
                
                next_location = next_visited_location['location']
                next_visited_location.pop('location', None)
                result = next_visited_location
                result_name = "{}->{}".format(pre_location, next_location)
                planning[result_name] = result

                pre_location = next_location
                pre_accumulated_time = result['arrivalTime'] + result['duration']

            positive_score = 0
            for route in planning:
                if planning[route]['score'] > 0:
                    positive_score += planning[route]['score']

            if positive_score > best_positive_score:
                best_positive_score = positive_score
                best_planning = planning

        return best_planning, best_positive_score

            # print(len(planning))
            # break


    def solve(self):
        """
        Brute force to find the best solution
        """ 
        print("Start to solve the problem!")

        planning_day = {i+1: None for i in range(int(self.max_days))}

        # Find best with 1 day
        best_positive_score = -1e5
        best_planning = None
        for home_location in self.location_arr:
            unvisited_locations = self.location_arr.copy()
            unvisited_locations.remove(home_location)

            planning, positive_score = self.solve_single(home_location, unvisited_locations)
            if positive_score > best_positive_score:
                best_positive_score = positive_score
                best_planning = planning
        
        for route in best_planning:
            home_location = int(route.split("->")[0])
            break 

        def find_arnomal_start_route(planning):
            # First element is negative or day transition
            arnomal_index = None
            for i, (key, value) in enumerate(planning.items()):
                if i == 0:
                    previous_arrivalTime = value['arrivalTime']
                    continue
                if value['score'] < 0 or value['arrivalTime'] < previous_arrivalTime:
                    return i

            return arnomal_index

        plan_for_day = 1
        current_planning = best_planning

        while plan_for_day <= self.max_days:
            arnomal_index = find_arnomal_start_route(current_planning)
            if arnomal_index is None:
                planning_day[plan_for_day] = current_planning 
                # IF WE NEED TO USE ALL DAY CAPACITY
                break

            all_loc2loc = list(current_planning.keys())
            day_planning = {key: current_planning[key] for key in all_loc2loc[:arnomal_index]}

            # back to home in the end of the day
            # rs_name = "{}->{}".format(all_loc2loc[arnomal_index-1].split("->")[1], home_location)
            # rs_value = {'arrivalTime': 1.43, 'duration': 2.84, 'cost': 654.660001, 'transport': 1, 'score': 38.25}


            unvisited_locations = [int(location.split("->")[1]) for location in all_loc2loc[arnomal_index:]]
            planning_day[plan_for_day] = day_planning

            current_planning, _ = self.solve_single(home_location, unvisited_locations)
            plan_for_day += 1


        # print(planning_day)
        final_planning = {}
        for day in planning_day:
            if planning_day[day] is None:
                break

            for loc2loc in planning_day[day]:
                loc2loc_info = planning_day[day][loc2loc]
                arrivalTime, duration, spend_money, transport_id = loc2loc_info['arrivalTime'], loc2loc_info['duration'], loc2loc_info['cost'], loc2loc_info['transport']
                final_planning[loc2loc] = [float(day), arrivalTime, duration, spend_money, transport_id]

                pre_location, pre_arrivalTime, pre_duration = int(loc2loc.split("->")[1]), arrivalTime, duration
            
            
            # back to home in the end of the day
            # BECAREFUL IF THE ARRIVALTIME IS HIHGER THAN 24H
            back2home_name = "{}->{}".format(pre_location, home_location)
            
            back2home_transport_id = 1
            back2home_distance = self.get_distance(pre_location, home_location) / 1000
            back2home_travelTime = back2home_distance / self.transport_csv[back2home_transport_id]['speed']
            # back2home_transport_cost = self.transport_csv[back2home_transport_id]['cost'] * back2home_distance
            # back2home_cost = (self.location_csv[home_location]['minPrice'] + back2home_transport_cost)* self.person_count
            back2home_cost = self.location_csv[home_location]['minPrice']* self.person_count

            back2home_value = [float(day), pre_arrivalTime + pre_duration + back2home_travelTime, self.user_arrival_duration_time[home_location]['best_duration'], back2home_cost, back2home_transport_id]
            final_planning[back2home_name] = back2home_value

        # add final line, not use to evaluate
        final_planning["{}->0".format(home_location)] = [final_planning[back2home_name][0], 21, 21, 21, 1]
        # print(self.evaluate(final_planning))
        # print("="*100)
        final_planning = self.fix_total_cost(final_planning)

        # print(self.evaluate(final_planning))
        submission = self.save_solution(final_planning)
        return submission
        

    def save_solution(self,planning):
        """
        Save solution to csv file
        """
        submission = pd.DataFrame(columns=['noDay', 'locationID', 'arrivalTime', 'duration', 'spendMoney', 'transportID'])
        for key, value in planning.items():
            no_day, arrival_time, duration, spend_money, transport_id = value
            location_id = key.split('->')[0]
            submission = submission.append({'noDay': int(no_day), 'locationID': int(location_id), 'arrivalTime': float(arrival_time), 'duration': float(duration), 'spendMoney': float(spend_money), 'transportID': int(transport_id)}, ignore_index=True)

        return submission


    def check_time_constraint(self, planning):
        threshold = 1e-3

        items = planning.items()
        day_evaluate = 1
        for i, (key, value) in enumerate(items):
            if value[0] == day_evaluate:
                pre_location = key.split('->')[1]
                _, pre_arrival_time, pre_duration, pre_spend_money, pre_transport_id = value
                day_evaluate += 1
                continue
            elif i == (len(items) - 1):
                continue

            no_day, arrival_time, duration, spend_money, transport_id = value
            cur_location = key.split('->')[1]
            pre_arrival = pre_arrival_time + pre_duration + self.get_distance(pre_location, cur_location) / 1000 / self.transport_csv[transport_id]['speed']
            
            
            if (arrival_time - pre_arrival) > threshold:
                print(pre_location, cur_location, arrival_time - pre_arrival)
                return False

            pre_location = cur_location
            pre_arrival_time, pre_duration, pre_transport_id = arrival_time, duration, transport_id

        return True       
        
    def evaluate_single_location(self, location_id, arrivalTime, duration, spend_money):
        score = 0
        #user preference

        score += 0.2 * np.linalg.norm(self.user_preference_location[location_id]) ** 2

        # Time arrival and duration
        def cal_timearrival_score(time, time_window):
            lower_time, upper_time = time_window
            if lower_time <= time <= upper_time:
                return 5
            elif time < lower_time:
                return -25 * (lower_time - time)
            elif time > upper_time:
                return -25 * (time - upper_time)

        for i in range(len(self.user_arrival_duration_time[location_id]["arrivalTime"])):
            arrival_time_window = self.user_arrival_duration_time[location_id]["arrivalTime"][i]
            duration_time_window = self.user_arrival_duration_time[location_id]["duration"][i]
            score += cal_timearrival_score(arrivalTime, arrival_time_window) + cal_timearrival_score(duration, duration_time_window)
        
        # Money 
        all_cost_per_user = spend_money / self.person_count
        # transport_cost_per_user = transport_cost / self.person_count
        if self.location_csv[location_id]['minPrice'] <= all_cost_per_user <= self.location_csv[location_id]['maxPrice']:
            score += self.person_count * 5
        else:
            score -= self.person_count * 10

        return score

    def evaluate(self, planning):
        
        if not self.check_time_constraint(planning):
            return 0

        total_score= 0 
        # cost evaluate
        minCost, maxCost = self.budget_csv['minWallet'], self.budget_csv['maxWallet']
        totalCost = 0
        for i, (key, value) in enumerate(planning.items()):
            if i == (len(planning) - 1):
                continue
            no_day, arrival_time, duration, spend_money, transport_id = value
            
            pre_location = int(key.split("->")[0])
            current_location = int(key.split("->")[1])

            transport_cost = self.transport_csv[transport_id]['cost'] * (self.get_distance(pre_location, current_location) / 1000) * self.person_count
            totalCost += float(spend_money) + transport_cost
    
            score = self.evaluate_single_location(current_location, arrival_time, duration, spend_money)
            print(key, score)
            total_score += score

        # print(totalCost, minCost, maxCost)
        if minCost <= totalCost <= maxCost:
            total_score += 50
        elif totalCost < minCost:
            total_score -= 400* (minCost - totalCost)/minCost
        elif totalCost > maxCost:
            total_score -= 400* (totalCost - maxCost)/maxCost

        
        
        return total_score




if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    src = "./data/QAIPublic/Q1/"
    budget_file = src + "budget.csv"
    location_file = src + "location.csv"
    roadtrip_file = src + "roadtrip.csv"
    transport_file = src + "transport.csv"
    usertrip_file = src + "usertrip.csv"


    usertrip = pd.read_csv(usertrip_file)
    # usertrip.arrivalTime = usertrip.arrivalTime.apply(literal_eval)
    # usertrip.duration = usertrip.duration.apply(literal_eval)

    budget = pd.read_csv(budget_file)
    location = pd.read_csv(location_file)

    # location = {x['locationID']: x for x in location}

    # location2id = {location[i]['locationID']: i for i in location}

    roadtrip = pd.read_csv(roadtrip_file)
    # roadtrip = {str(x['Unnamed: 0']): x for x in roadtrip}


    transport = pd.read_csv(transport_file)
    # transport = {x['transportationID']: x for x in transport}

    # location_arr = usertrip['locationID'].unique() 
    # person_count = len(usertrip['userName'].unique())


    # print(roadtrip['1737']['1859'])

    mysolver = Solver(usertrip, location, roadtrip, transport, budget)


    submission = mysolver.solve()
    submission.to_csv('submission.csv', index=False)
    print(submission)
    # location_time_arrive = []
    # for location_list in location_factorial:
    #     location_list = list(location_list)
        
    #     # home_location = location_list[0]

    #     # location_list.remove(home_location)

        
    #     for location_id in location_list:
    #         usertrip_filter = usertrip[usertrip['locationID'] == location_id]
    #         arrival_time_list = usertrip_filter['arrivalTime']
    #         duration_time_list = usertrip_filter['duration']
    #         location_time_arrive.append((mysolver.best_fitness(arrival_time_list), mysolver.best_fitness(duration_time_list)))

    #     break

    # location_time_arrive.sort(key= lambda x: x[0])
    # print(location_time_arrive)
