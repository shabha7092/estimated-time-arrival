# estimated-time-arrival
This machine learning module predicts the estimated time arrival for order delivery.
The whole module is configured in a docker container which predicts the estimated time's. It also produces various plot's for model validation 

* Scikit
* Pandas
* Fractional Stratification
* DataBricks
* Docker

Execution Command:
docker build -t estimated-time-arrival . && docker run -v $(pwd)/Output:/app/Output -it estimated-time-arrival
