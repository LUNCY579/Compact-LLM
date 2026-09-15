#!/usr/bin/env bash
# Retired operational wrapper; original bytes are retained in the historical source.
printf '%s\n' 'REFUSED: this MQ wrapper bypassed supervision or deleted checkpoint state.' >&2
printf '%s\n' 'Use python -m experiments.model_quality_gpu_v1 train from the qualified successor checkout. Retain all checkpoints.' >&2
exit 2
