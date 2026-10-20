#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
# Disable every project opt-in, including external fixtures, on inherited/self-hosted environments.
# Preserve only the explicit build-engine choice. This command never enables native qualification.
for name in $(env | sed -n 's/^\(AETHER_[A-Z0-9_]*\)=.*/\1/p'); do
    case "$name" in AETHER_BUILD_SYSTEM) ;; *) unset "$name" ;; esac
done
exec sh Scripts/check.sh
