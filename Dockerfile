FROM ultralytics/ultralytics
RUN pip install flask

WORKDIR /app

COPY . /app

EXPOSE 5000
EXPOSE 8080
EXPOSE 80 443 22
EXPOSE 7000-8000

CMD ["python", "app.py" , "-video", "./sample.mp4"]

