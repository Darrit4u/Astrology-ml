FROM python:3.9
# ENV PYTHONUNBUFFERED 1
RUN mkdir /astrology
WORKDIR /astrology
RUN git clone https://github.com/FoxLand-b4n/Astrology-ml.git
RUN pip install -r ./Astrology-ml/requirements.txt
CMD ["python", "Astrology-ml/astrology_site/manage.py runserver localhost:8000"]