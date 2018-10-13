VERSION=1.1
TRAIN_IMAGE_BASE=alvinhenrick/kube-demo-dist
SERVE_IMAGE_BASE=alvinhenrick/imdb-classification

build:
	docker build -t ${TRAIN_IMAGE_BASE}:${VERSION} .

login:
	docker login

push:
	docker push ${TRAIN_IMAGE_BASE}:${VERSION}

train:
	kubectl apply -f tfjobdist.yaml

download:
	kubectl cp dataaccess:/model/imdb_model ./imdb_model

s2i:
	s2i build . seldonio/seldon-core-s2i-python3:0.1 ${SERVE_IMAGE_BASE}:${VERSION} --copy ./imdb_model --env MODEL_NAME=ImdbClassifier --env API_TYPE=REST --env SERVICE_TYPE=MODEL --env PERSISTENCE=0

s2ipush:
	docker push ${SERVE_IMAGE_BASE}:${VERSION}

serve:
	cd dist_demo_ks ; ks generate seldon-serve-simple imdb-classification --image=${SERVE_IMAGE_BASE}:${VERSION}
	cd dist_demo_ks ; ks apply default -c imdb-classification

portforward:
	kubectl port-forward `kubectl get pods -n default -l service=ambassador -o jsonpath='{.items[0].metadata.name}'` -n default 8080:80

predict:
	curl -X POST -H 'Content-Type: application/json' -d '{"data":{"ndarray":[["I really liked the movie!"],["I hated the movie..."]]}}' http://localhost:8080/seldon/imdb-classification/api/v0.1/predictions

tail:
	kubectl logs -f kube-demo-dist-master-0

tailseldon:
	kubectl logs -f `kubectl get pods -l seldon-app=imdb-classification -o=jsonpath='{.items[0].metadata.name}'` imdb-classification

stop:
	kubectl delete -f tfjobdist.yaml

clean:
	cd dist_demo_ks ; ks delete default -c imdb-classification
	cd dist_demo_ks ; ks component rm imdb-classification


