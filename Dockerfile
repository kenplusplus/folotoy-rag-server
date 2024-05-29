FROM python:3.11.5-slim-bullseye

COPY . /src

WORKDIR /src

RUN pip install -r requirements.txt

ENV OPENAI_API_KEY="EMPTY"
ENV OPENAI_BASE_URL="http://101.201.111.141:8000/v1"

EXPOSE 8001

CMD ["python3", "app.py"]