#!/bin/sh
set -e
cd "$(dirname "$0")"
./kill_client.sh
./wipe_server.sh
./reset_client.sh
