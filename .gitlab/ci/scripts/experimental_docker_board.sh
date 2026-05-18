#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-run}"
: "${BOARD_USER:=tensorrt_user}"
: "${BOARD_IP:=192.168.55.1}"
: "${REMOTE_WORKSPACE:=/home/${BOARD_USER}/tensorrt-edge-llm-docker-${CI_PIPELINE_ID:-manual}-${CI_JOB_ID:-local}}"
: "${EXPERIMENTAL_DOCKER_IMAGE:=tensorrt-edge-llm:experimental-${CI_COMMIT_SHORT_SHA:-local}}"
: "${CUTE_DSL_ARTIFACT_NAME:=cutedsl_aarch64_sm_110_cuda13.tar.gz}"
: "${CUTE_DSL_PREBUILT_DIR:=kernelSrcs/cuteDSLPrebuilt}"

require_env() {
  if [ -z "${!1:-}" ]; then
    echo "$1 is required"
    exit 1
  fi
}

ssh_to_board() {
  sshpass -p "$BOARD_PASSWORD_NVKS" \
    ssh -o StrictHostKeyChecking=no "$BOARD_USER@$BOARD_IP" "$@"
}

remote_bash() {
  ssh_to_board \
    "SUDO_PASS=${BOARD_PASSWORD_NVKS@Q} EXPERIMENTAL_DOCKER_IMAGE=${EXPERIMENTAL_DOCKER_IMAGE@Q} REMOTE_WORKSPACE=${REMOTE_WORKSPACE@Q} bash -se"
}

stage_cutedsl_artifact() {
  mkdir -p "$CUTE_DSL_PREBUILT_DIR"

  if [ -f "$CUTE_DSL_ARTIFACT_NAME" ]; then
    mv -f "$CUTE_DSL_ARTIFACT_NAME" "$CUTE_DSL_PREBUILT_DIR/"
  fi
  if [ -f "$CUTE_DSL_ARTIFACT_NAME.sha256" ]; then
    mv -f "$CUTE_DSL_ARTIFACT_NAME.sha256" "$CUTE_DSL_PREBUILT_DIR/"
  fi

  if [ ! -f "$CUTE_DSL_PREBUILT_DIR/$CUTE_DSL_ARTIFACT_NAME" ]; then
    echo "Missing CuteDSL artifact: $CUTE_DSL_PREBUILT_DIR/$CUTE_DSL_ARTIFACT_NAME"
    echo "The Docker job must download build_cutedsl_sm110_artifact before building."
    exit 1
  fi

  if [ -f "$CUTE_DSL_PREBUILT_DIR/$CUTE_DSL_ARTIFACT_NAME.sha256" ]; then
    (cd "$CUTE_DSL_PREBUILT_DIR" && sha256sum -c "$CUTE_DSL_ARTIFACT_NAME.sha256")
  fi
}

sync_workspace() {
  ssh_to_board "mkdir -p '$REMOTE_WORKSPACE'"
  sshpass -p "$BOARD_PASSWORD_NVKS" \
    rsync -az --delete --exclude .git -e "ssh -o StrictHostKeyChecking=no" \
    ./ "$BOARD_USER@$BOARD_IP:$REMOTE_WORKSPACE/"
}

run_on_board() {
  remote_bash <<'BOARD_SCRIPT'
set -euo pipefail

sudo_run() {
  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' "$@"
}

cd "$REMOTE_WORKSPACE"
echo "Board hostname: $(hostname)"
sudo_run docker version
sudo_run docker build --network=host --shm-size=8g -f experimental/docker/Dockerfile -t "$EXPERIMENTAL_DOCKER_IMAGE" .
sudo_run docker run --runtime nvidia --rm --network host --shm-size=8g "$EXPERIMENTAL_DOCKER_IMAGE" python3 -m experimental.server --help
sudo_run docker run --runtime nvidia --rm --network host --shm-size=8g "$EXPERIMENTAL_DOCKER_IMAGE" python3 -c 'from experimental.server.engine import _import_runtime; print(_import_runtime().__name__)'
BOARD_SCRIPT
}

cleanup_board() {
  remote_bash <<'BOARD_SCRIPT'
set -euo pipefail

sudo_run() {
  printf '%s\n' "$SUDO_PASS" | sudo -S -p '' "$@"
}

if command -v docker >/dev/null 2>&1; then
  container_ids="$(sudo_run docker ps -aq --filter "ancestor=$EXPERIMENTAL_DOCKER_IMAGE" || true)"
  if [ -n "$container_ids" ]; then
    sudo_run docker rm -f $container_ids || true
  fi
  sudo_run docker rmi "$EXPERIMENTAL_DOCKER_IMAGE" || true
fi

rm -rf "$REMOTE_WORKSPACE" || true
BOARD_SCRIPT
}

require_env BOARD_PASSWORD_NVKS

case "$ACTION" in
  run)
    echo "Board target: ${BOARD_USER}@${BOARD_IP}"
    echo "Remote workspace: ${REMOTE_WORKSPACE}"
    stage_cutedsl_artifact
    sync_workspace
    run_on_board
    ;;
  cleanup)
    cleanup_board
    ;;
  *)
    echo "Unknown action: $ACTION"
    exit 2
    ;;
esac
