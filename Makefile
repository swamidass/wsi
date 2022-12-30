.PHONY: docker-base docker-dev

DOCKER_REPO?=dockerreg01.accounts.ad.wustl.edu/swamidass/

all: docker-dev

docker-dev: environment.yml requirements.txt
	docker build -f Dockerfile . -t wsi:dev


push:
	docker tag wsi:dev ${DOCKER_REPO}wsi:dev
	docker push ${DOCKER_REPO}wsi:dev


shell:
	docker run -it --rm wsi:dev


clean:
	docker image prune -f
	docker builder prune -f
