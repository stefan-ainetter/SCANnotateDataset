#!/bin/bash
DIR_PATH=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
file="visualize_annotations.py"
execute_path=$DIR_PATH/$file

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

echo "gpu = $gpu"

python3 $execute_path --device=$gpu