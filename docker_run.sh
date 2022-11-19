docker run --network host -it --rm -v `pwd`:/scratch --user $(id -u):$(id -g) featurescoop:latest bash -c \
    "(cd /scratch/featurescoop && python api.py)"