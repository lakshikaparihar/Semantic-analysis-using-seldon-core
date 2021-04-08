FROM continuumio/miniconda3

RUN pip3 install --upgrade pip
RUN pip3 install mlflow \
    && pip3 install pandas \
    && pip3 install scikit-learn \
    && pip3 install scipy 
ADD requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install pymysql
RUN export MLFLOW_TRACKING_URI=mysql+pymysql://nishkarsh:nishkarsh@nishkarsh.cmuzdwd6qin5.ap-south-1.rds.amazonaws.com:3306/nishkarsh
