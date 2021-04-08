#!/usr/bin/env sh

echo "Deploying Production model name=MovieModel"

# Set enviorment variable for the tracking URL where the Model Registry is
export MLFLOW_TRACKING_URI=mysql+pymysql://nishkarsh:nishkarsh@nishkarsh.cmuzdwd6qin5.ap-south-1.rds.amazonaws.com:3306/nishkarsh
# Serve the production model from the model registry
mlflow models serve --model-uri models:/MovieModel/production --no-conda -p 3000
