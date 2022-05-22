FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3" , "main.py"]
