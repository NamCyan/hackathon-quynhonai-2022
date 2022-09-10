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

import logging
import pandas as pd

class Evaluation():
    '''
    Evaluating the submission
    '''
    def __init__(self, path):
        self.path = path
        self.usertrip_csv = pd.read_csv(self.path + 'usertrip.csv')
        self.location_csv = pd.read_csv(self.path + 'location.csv')
        self.budget_csv = pd.read_csv(self.path + 'budget.csv')
        self.roadtrip_csv = pd.read_csv(self.path + 'roadtrip.csv')
        self.transport_csv = pd.read_csv(self.path + 'transport.csv')
        self.submission_csv = pd.read_csv(self.path + 'submission.csv')  
        self.total_people = len(set(self.usertrip_csv['userID']))
        self.location_list = self.submission_csv['locationID'].tolist()
        self.home_location = self.location_list[0]

    def get_distance(self, start_location, end_location):
        '''
        Get distance between two locations
        '''
        start_location, end_location = int(start_location), int(end_location)
        first_column = self.roadtrip_csv.iloc[:,0]
        index = first_column.index[first_column == start_location]
        value = self.roadtrip_csv[str(end_location)].iloc[index].values[0]
        return value

    def check_twice_visited(self):
        '''
        Check if a location is visited twice
        '''
        revisited_location = [x for x in self.location_list if x != self.home_location] # Ignore the first location
        if len(revisited_location) != len(set(revisited_location)):
            return False
        return True

    def consistent_logic_check(self):
        '''
        Check if the submission is consistent with the input
        '''
        for index, row in self.submission_csv.iterrows():
            transport_id = int(row['transportID'])
            transport_speed = self.transport_csv.loc[self.transport_csv['transportationID'] == transport_id]['speed'].values[0] * 1000 # km/h to m/h
            location_id = int(row['locationID'])
            arrival_time = row['arrivalTime']
            if index != 0 and index != self.submission_csv.shape[0] - 1:
                prev_location = self.submission_csv.iloc[index - 1]
                next_location = self.submission_csv.iloc[index + 1]
                next_location_id = int(next_location['locationID'])
                prev_arrival_time = prev_location['arrivalTime']
                prev_duration_time = prev_location['duration']
                if row['noDay'] == prev_location['noDay']:
                    distance = self.get_distance(location_id, next_location_id)
                    time_prediction = arrival_time - prev_duration_time - (distance / transport_speed)
                    if (prev_arrival_time - time_prediction)**2 > 0.08: 
                        sen = "Fail to check time at " + str(next_location_id) + " your time prediction: "
                        sen += str(arrival_time) + " Actual time: "
                        sen += str(prev_arrival_time + prev_duration_time + (distance / transport_speed))
                        logger.info(sen)
                        return False
        return True
    
    def get_budget(self):
        '''
        Get total of spend money cost
        '''
        total_cost = self.submission_csv['spendMoney'].sum()
        total_cost -= self.submission_csv.iloc[-1]['spendMoney']
        cost_list = []
        for index, row in self.submission_csv.iterrows():
            if index != len(self.submission_csv) -1:
                transport_id = int(row['transportID'])
                transport_cost = self.transport_csv[self.transport_csv['transportationID'] == transport_id]['cost'].values[0]
                location_id = row['locationID']
                next_location_id = self.submission_csv.iloc[index + 1]['locationID']
                distance = self.get_distance(location_id, next_location_id) / 1000
                transport_cost *= distance
                transport_cost *= self.total_people
                cost_list.append(transport_cost)
        total_cost += sum(cost_list)
        return total_cost
    
    def section_a_evaluation(self):
        '''
        Calcuate the score of section A
        '''
        point = 0
        
        for index, row in self.submission_csv.iterrows():
            if index != len(self.submission_csv) -1:
                location_cost = row['spendMoney']
                next_location_id = self.submission_csv.iloc[index + 1]['locationID']
                min_price = self.location_csv[self.location_csv['locationID'] == next_location_id]['minPrice'].values[0] 
                min_price *= self.total_people
                max_price = self.location_csv[self.location_csv['locationID'] == next_location_id]['maxPrice'].values[0] 
                max_price *= self.total_people
                output_log = f"Correct budget constraint at location {next_location_id} with budget {location_cost} " 
                output_log += f"respect to min price {min_price} and max price {max_price}"
                if min_price <= location_cost <= max_price:
                    point += 5
                    # logger.info(output_log) 
                    # print(next_location_id)
                else:
                    point -= 10
                    output_log = output_log.replace("Correct", "Wrong")
                    # print(next_location_id)
                    # logger.warning(output_log)
                    
        return point

    def section_b_evaluation(self):
        '''
        Calcuate the score of section B
        '''
        point = 0
        total_cost = self.get_budget()
        max_wallet = self.budget_csv['maxWallet'].values[0]
        min_wallet = self.budget_csv['minWallet'].values[0]
        if total_cost > max_wallet:
            point -= ((total_cost / max_wallet) - 1) * 400
        elif total_cost < min_wallet:
            point -= (1 - (total_cost / min_wallet)) * 400
        else:
            point += 50
        return point
    
    def section_c_evaluation(self):
        '''
        Calcuate the score of section C
        '''
        point = 0
        for key, user in self.usertrip_csv.iterrows():
            user_location_id = user['locationID']
            user_preference = user['preference']
            if user_location_id in self.location_list and user_location_id != self.home_location:
                point += 0.2 * user_preference ** 2
        
        return point

    def section_d_evaluation(self):
        '''
        Calcuate the score of section D
        '''
        point = 0
        for key, user in self.usertrip_csv.iterrows():
            user_location_id = user['locationID']
            user_arrival_time = list(map(float, user['arrivalTime'][1:-1].split(',')))
            user_duration_time = list(map(float, user['duration'][1:-1].split(',')))
            if user_location_id in self.location_list and user_location_id != self.home_location:

                # print(user_location_id)
                location_index = self.submission_csv[self.submission_csv['locationID'] == user_location_id].index.values[0]
                prev_location = self.submission_csv.iloc[location_index - 1]
                arrival_time = prev_location['arrivalTime']
                duration_time = prev_location['duration']

                def get_point(time,time_window):
                    small_point = 0
                    if time_window[0] <= time <= time_window[1]:
                        small_point += 5
                    elif time > time_window[1]:
                        small_point -= (time - time_window[1]) * 25
                    else:
                        small_point -= (time_window[0] - time) * 25
                    return small_point
                
                point += get_point(arrival_time, user_arrival_time)
                point += get_point(duration_time, user_duration_time)
        return point



    def display(self,section,point):
        '''
        Display the score of each section
        '''
        logger.info('-'*50)
        logger.info(section)
        logger.info(f'Point of {section}: {point}')
        logger.info('-'*50)       

    def score_evaluation(self):
        '''
        Evaluating the score
        '''
        if not self.check_twice_visited():
            logger.warning('Invalid submission format, location is visited twice')
            return 0       
        if not self.consistent_logic_check():
            logging.info("Incorrect time between arrivalTime and nextArrivalTime")
            return 0   
        # Section a
        section_a_point = self.section_a_evaluation()
        print('Section a',section_a_point)
        # Section b
        section_b_point = self.section_b_evaluation()
        print('Section b',section_b_point)
        # Section c
        section_c_point = self.section_c_evaluation()
        print('Section c',section_c_point)
        # Section d
        section_d_point = self.section_d_evaluation()
        print('Section d',section_d_point)

        total_score = section_a_point + section_b_point + section_c_point + section_d_point

        return total_score
               
if __name__ == '__main__':
    logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    # file_handler = logging.FileHandler('eval.log',mode='w')
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    # logger.info('Start evaluating')
    PATH = './data/Q1/'

    evaluation = Evaluation(PATH)
    SCORE = evaluation.score_evaluation()

    print(SCORE)

    