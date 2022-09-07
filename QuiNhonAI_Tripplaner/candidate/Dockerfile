FROM python:3.9.10

COPY new_solution.py /app/new_solution.py
COPY requirements.txt /app/requirements.txt
COPY api.py /app/api.py

WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "api.py"]