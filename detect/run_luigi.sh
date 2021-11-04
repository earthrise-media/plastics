#!/bin/bash

# Check ENV
echo "Preflight checks...."
missing_env=()
if [[ -z "${AWS_ACCESS_KEY_ID}" ]]; then
  missing_env+=("AWS_ACCESS_KEY_ID")
fi
if [[ -z "${AWS_SECRET_ACCESS_KEY}" ]]; then
  missing_env+=("AWS_SECRET_ACCESS_KEY")
fi
if [[ -z "${DESCARTESLABS_CLIENT_ID}" ]]; then
  missing_env+=("DESCARTESLABS_CLIENT_ID")
fi
if [[ -z "${DESCARTESLABS_CLIENT_SECRET}" ]]; then
  missing_env+=("DESCARTESLABS_CLIENT_SECRET")
fi
missing_length=${#missing_env[@]}
if (( $missing_length > 0  )); then
  echo "Preflight checks failed - the following ENV vars are missing:"
  printf '%s\n' "${missing_env[@]}"
else
  echo "Preflight checks passed...starting now"
  echo " "

  luigi "$@"
fi