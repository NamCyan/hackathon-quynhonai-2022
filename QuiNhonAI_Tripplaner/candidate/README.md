## Trip Planner QAI Hackathon

### Project Structure
```bash
candidate
│   api.py                        # This module used for expose API (keep as template)
│   Dockerfile                    # Text document to build docker image
│   requirements.txt              # All libraries need to be installed in docker image
|   sample_solution.py            # This is a sample solution (Please replace your solution)
└───dataset                       # Dataset is avaiable on QAI Hackathon Website
    └───Q1                        # Public dataset
        |   budget.csv            # Contains budget information
        │   location.csv          # Contains location information
        │   roadtrip.csv          # Contains distance matrix information
        │   transport.csv         # Contains transport information
        |   usertrip.csv          # Contains user trip information
    └───Q2...
```

### Guideline for builidng docker image and upload to S3

#### 1) Build your docker image (Dockerfile)

```bash
docker build -t your_docker_username/qai-tripplanner:latest .
# For example
docker build -t qaifsoft/qai-tripplanner:latest .
```

#### 2) Check your dokcer image after building

```bash
docker images | grep 'qai-tripplanner'
```

#### 3) Run your created docker image (run docker image)

```bash
# Example script: docker run --name container_name -p port:port -d your_docker_username/qai-tripplanner:latest
docker run --name trip_planner -p 5000:5000 -d qaifsoft/qai-tripplanner:latest
```

Notice: If you have a response error like that: 

```bash
docker: Error response from daemon: Conflict. The container name "/trip_planner" is already in use by container "xxxxxx". You have to remove (or rename) that container to be able to reuse that name.
```

You can use the following command to remove the container:

```bash
docker rm -f trip_planner
```

#### 4) Wrap your docker image to docker-image.tar

```bash
docker save -o docker-image.tar qaifsoft/qai-tripplanner:latest
```

#### 5) Upload your docker image to QAI Hackthon Submission page

### Guideline for run your solution

```bash
python3 sample_solution.py
```

The sample_submission folder will be created after running your solution. Please check your submission.CSV format before submitting to QAI Hackathon Submission page. 

#### Sample submission.csv format

```csv
noDay,locationID,arrivalTime,duration,spendMoney,transportID
1,467,2.21,0.24,110.0,1 # locationID 467 is HOME
1,293,3.20,0.5,140.0,1
1,1192,6.5,1,200.0,1
2,467,7.7,2,300.0,1 # You must come back to HOME location after each day
2,2051,14.20,2,400.0,1
2,467,17.21,0.24,110.0,1 # Comeback to HOME location when you finish the journey
```

Transcript:
1. 467 is HOME location
2. From the first row, you can see that the tourist will arrive at location 293 at 2.21 AM and spend 0.24 hours at this location and use transportID 1 to go to location 293 from 467. Remember that, arrivalTime 2.21 contains the travel time.
3. From the second row, the tourist will arrive at location 1192 at 3.20 AM, spend 0.5 hours at this location and use transportID 1 to go to location 1192 from 293.

Another example:

```csv
noDay,locationID,arrivalTime,duration,spendMoney,transportID
1,320,9.0,1.5,300.0,1.0
1,2061,10.9,0.5,150.0,4.0
1,2069,11.42,1.2,0.0,1.0
1,2071,15.55,0.7,500.0,4.0
2,320,9.0,1.0,160.0,1.0
2,486,10.13,2.0,3000.0,1.0
2,931,12.17,1.5,1000.0,1.0
2,957,14.07,2.0,1400.0,1.0
2,320,14.57,1.0,600.0,1.0
```

To be clear, please see our explanation for the format of the submission.CSV file.

![](https://i.imgur.com/CsS5qL6.png)

All data type in submission.csv must be float or int. Default, the first locationID is 'HOME' location. Therefore, you must comeback to 'HOME' location after each day and the end of the journey. 