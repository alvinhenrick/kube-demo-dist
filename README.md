#TODO STEPS

### Tail Log
`kubectl logs -f $(kubectl get pods -l seldon-app=imdb-classification -o=jsonpath='{.items[0].metadata.name}') imdb-classification`

### Port Forward
`kubectl port-forward $(kubectl get pods -n default -l service=ambassador -o jsonpath='{.items[0].metadata.name}') -n default 8080:80`