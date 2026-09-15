#!/bin/zsh

script_dir=${0:A:h}
cd "$script_dir" || exit 1

print "Starting the DwarfStar model supervisor for Pi GUI."
print "Pi's model picker will unload and replace the active model automatically."
print "API: http://${DS4_HOST:-100.109.208.12}:${DS4_API_PORT:-8000}/v1"
print "Control: http://${DS4_HOST:-100.109.208.12}:${DS4_CONTROL_PORT:-8001}"
print "Press Ctrl-C to stop the supervisor and unload the model."

exec /usr/bin/python3 scripts/ds4_model_supervisor.py \
  --host "${DS4_HOST:-100.109.208.12}" \
  --api-port "${DS4_API_PORT:-8000}" \
  --control-port "${DS4_CONTROL_PORT:-8001}" \
  --initial "${DS4_INITIAL_PROFILE:-deepseek-v4}" \
  --token "${DS4_CONTROL_TOKEN:-dsv4-local-switch}"
