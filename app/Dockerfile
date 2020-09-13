FROM python:3.7.9-slim
WORKDIR /opt
RUN python -m pip install flask
RUN python -m pip install tensorflow
RUN python -m pip install pandas
RUN python -m pip install youtube-transcript-api
COPY . .
EXPOSE 5000
CMD ["python", "/opt/application.py"]