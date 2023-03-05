#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DEVICE=0
echo "Using GPU devices: ${DEVICE}"


docker run \
    -it --rm \
    --name "xaicyclegan" \
    --gpus all \
    --privileged \
    --shm-size 8g \
    -v "${HOME}/.netrc":/root/.netrc \
    -v "${CWD}/..":/workspace/${PROJECT_NAME} \
    -v "/mnt/scratch/${USER}/${PROJECT_NAME}":/workspace/${PROJECT_NAME}/.mnt/scratch \
    -v "/mnt/persist/${USER}/${PROJECT_NAME}":/workspace/${PROJECT_NAME}/.mnt/persist \
    -e CUDA_VISIBLE_DEVICES="${DEVICE}" \
    ${IMAGE_TAG} \
    "$@" || exit $?
