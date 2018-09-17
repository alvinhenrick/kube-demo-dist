.PHONY: build

build:
	docker build -t alvinhenrick/kube-demo-dist .

login:
	docker login

push:
	docker push alvinhenrick/kube-demo-dist

train:
	kubectl apply -f tfjobdist.yaml

download:
	kubectl cp dataaccess:/model/imdb_model ./imdb_model

s2i:
	s2i build . seldonio/seldon-core-s2i-python3:0.1 alvinhenrick/imdb-classification:0.1 --env MODEL_NAME=ImdbClassifier --env API_TYPE=REST --env SERVICE_TYPE=MODEL --env PERSISTENCE=0

s2ipush:
	docker push alvinhenrick/imdb-classification:0.1

serve:
	ks generate seldon-serve-dist imdb-classification --image=alvinhenrick/imdb-classification:0.1
	ks apply default -c imdb-classification

portforward:
	kubectl port-forward `kubectl get pods -n default -l service=ambassador -o jsonpath='{.items[0].metadata.name}'` -n default 8080:80

predict:
	curl -X POST -H 'Content-Type: application/json' -d '{"data":{"ndarray":[['I really liked the movie!'],['Hated every second of it...']]}}' http://localhost:8080/seldon/imdb-classification/api/v0.1/predictions

clean:
	kubectl delete -f tfjobdist.yaml
	ks delete default -c imdb-classification
	ks component rm imdb-classification


