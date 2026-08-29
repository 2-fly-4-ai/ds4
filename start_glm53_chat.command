#!/bin/zsh

script_dir=${0:A:h}
cd "$script_dir" || exit 1

exec ./ds4-agent \
  -m gguf/GLM-5.3-Flash-Q2-Q4K-Attention.gguf \
  --ctx 16384 \
  --mtp --mtp-draft 2 \
  -sys "You are a helpful general-purpose assistant."
