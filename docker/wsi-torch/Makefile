.PHONY: docker-base docker-dev

DOCKER_REPO?=swamidass/
HASH=`git rev-parse --short HEAD`
DATE=torch-`date -I`

all: docker-dev

docker-dev: environment.yml requirements.txt
	echo ${HASH}
	docker build -f Dockerfile . -t wsi:dev

push:
	docker tag wsi:dev ${DOCKER_REPO}wsi:dev
	docker push ${DOCKER_REPO}wsi:dev
	
	docker tag wsi:dev  ${DOCKER_REPO}wsi:$(HASH)
	docker push ${DOCKER_REPO}wsi:$(HASH)
	docker rmi ${DOCKER_REPO}wsi:$(HASH)
	
	docker tag wsi:dev  ${DOCKER_REPO}wsi:$(DATE)
	docker push ${DOCKER_REPO}wsi:$(DATE)
	docker rmi ${DOCKER_REPO}wsi:$(DATE)

shell:
	docker run --shm-size=10gb -it --rm -v `pwd`:/var/task wsi:dev


clean:
	docker image prune -f
	docker builder prune -f
