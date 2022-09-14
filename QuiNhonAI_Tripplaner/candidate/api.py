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

import zipfile
import pandas as pd
from solution import Solver
from flask import Flask, request, Response, send_file

app = Flask(__name__)


# Health-checking method
@app.route('/healthCheck', methods=['GET'])
def health_check():
    """
    Health check the server
    Return:
    Status of the server
        "OK"
    """
    return "OK"


@app.route('/infer', methods=['POST'])
def infer():
    """
    Open API for inference
    """
    try:
        # Read file zip from request
        file = request.files['tripplan']
        # Save file to local
        file.save('/tmp/tripplan.zip') 
        list_file = []
        # Unzip file
        with zipfile.ZipFile('/tmp/tripplan.zip', 'r') as zip_ref:
            zip_ref.extractall('/tmp/data_inference')
            # Get listof file in folder
            list_file = zip_ref.namelist()
            # Get file name
        # Find usertrip name in list_file
        for file in list_file:
            url = '/tmp/data_inference/' + file
            if 'usertrip.csv' in file:
                usertrip_csv = pd.read_csv(url)
            if 'budget.csv' in file:
                budget_csv = pd.read_csv(url)
            if 'location.csv' in file:
                location_csv = pd.read_csv(url)
            if 'roadtrip.csv' in file:
                roadtrip_csv = pd.read_csv(url)
            if 'transport.csv' in file:
                transport_csv = pd.read_csv(url)             
        # # Combine all to one node
        # location_arr = usertrip_csv['locationID'].unique()
        # person_count = len(usertrip_csv['userName'].unique())
        
        # # WARNING: This is a sample solution, it is not optimized for performance 
        # # This approach is not scalable for large location list 
        # # Please optimize the solution to improve the performance 
        # # Don't try to use itertools.permutations, it will cause memory error  (if locations > 10) 
        # # And your submission return 0 score 
        # # To test the solution, we will only create 10 permutations

        # location_factorial = []

        # for index, location_arr in enumerate(itertools.permutations(location_arr)):
        #     location_factorial.append(location_arr)
        #     if index == 10: # 10 permutations
        #         break
        
        # This solver is force brute force, it can be optimized
        # Please replace by your algorithm
        mysolver = Solver(usertrip_csv, location_csv, roadtrip_csv, transport_csv, budget_csv)
        submission_csv = mysolver.solve()
        submission_csv.to_csv('/tmp/submission.csv', index=False)       
        # Response submission.csv
        return send_file("/tmp/submission.csv", as_attachment=True)

    except Exception:
        return Response(response='Invalid content type', status=500)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
