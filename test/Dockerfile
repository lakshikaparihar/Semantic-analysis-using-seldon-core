FROM seldonio/seldon-core-s2i-python3:1.6.0
RUN pip install --upgrade pip
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD stopwords.py .
RUN python stopwords.py
