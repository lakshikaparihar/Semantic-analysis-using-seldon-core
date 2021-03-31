#!/usr/bin/env sh

echo "Predicting Sentiment"
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json; format=pandas-records' -d   '{"data": {"ndarray": ["Veer-Zara is a great movie"]}}'
echo ""
echo "Done!"


