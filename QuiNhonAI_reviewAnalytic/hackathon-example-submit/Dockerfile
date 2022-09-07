# STEP 1: Pull python image
FROM python:3.9.13

# STEP 2,3: CREATE WORK DIR AND COPY FILE TO WORK DIR
WORKDIR /namlh_quinhonai
COPY requirements.txt /namlh_quinhonai
# COPY vi-roberta-base-detection/ /namlh_quinhonai
COPY vi-roberta-large-ensemble-PSUM-weighted/ /namlh_quinhonai

# STEP 4,5,6: INSTALL NECESSARY PACKAGE
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip install gdown

# STEP 7: Download file weight
# RUN gdown "https://drive.google.com/uc?export=download&id=1wRXCmjhOfM3fzh3hBufv5_aQx0j2Eomz"
# RUN unzip vi-roberta-large-mix-PSUM-256.zip
# RUN rm vi-roberta-large-mix-PSUM-256.zip

# RUN gdown "https://drive.google.com/uc?export=download&id=1tBFPtD9l51P4HQmjsnfZFkY5Sr-mMwpV"
# RUN unzip vi-roberta-base-detection.zip
# RUN rm vi-roberta-base-detection.zip

# STEP 8: RUN COMMAND
COPY . /namlh_quinhonai
CMD ["python", "./app.py"]