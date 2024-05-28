
#!/usr/bin/env bash

BASH_OPTION=bash

IMG=iscilab/graspnet:cuda-20-04
containerid=$(docker ps -qf "ancestor=${IMG}") && echo $containerid

xhost +

if [[ -n "$containerid" ]]
then
    docker exec -it \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e LINES="$(tput lines)" \
        henry_pybullet_module \
        $BASH_OPTION
else
    docker start -i henry_pybullet_module
fi
