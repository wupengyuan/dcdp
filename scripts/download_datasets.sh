#!/usr/bin/env bash
set -euo pipefail

datasets=("$@")
if [ "${#datasets[@]}" -eq 0 ]; then
  datasets=("pusht")
fi

if [ "${datasets[0]}" = "all" ]; then
  datasets=("pusht" "robomimic_lowdim" "kitchen")
fi

mkdir -p data

download_dataset() {
  local name="$1"
  local url="$2"
  local archive="data/${name}.zip"

  if [ -d "data/${name}" ]; then
    echo "Skipping ${name}: data/${name} already exists."
    return
  fi

  wget -O "${archive}" "${url}"
  unzip -q -o "${archive}" -d data
  rm -f "${archive}"
}

for dataset in "${datasets[@]}"; do
  case "${dataset}" in
    pusht)
      download_dataset pusht https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
      ;;
    robomimic_lowdim)
      download_dataset robomimic_lowdim https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip
      ;;
    kitchen)
      download_dataset kitchen https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip
      ;;
    *)
      echo "Unknown dataset: ${dataset}" >&2
      exit 1
      ;;
  esac
done
