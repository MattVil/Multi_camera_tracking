run:
	python run.py
debug:
	python debug.py
docker-run:
	./scripts/createContainer.sh \
		-i nvidia/tf-c-1.14-py3:1.0.1 \
		-d /home/depinfo/Projects/matthieu/ \
		-n mattcontainer
docker-exec:
	./scripts/launch.sh mattcontainer
