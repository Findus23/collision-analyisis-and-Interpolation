#!/usr/bin/env bash

set -x

targetdir="$HOME/tmp/cli"
rm -r $targetdir
mkdir -p $targetdir

cp "cli.py" $targetdir
cp "save.jsonl" $targetdir
cp "CustomScaler.py" $targetdir
cp -r "interpolators" $targetdir
cp "simulation.py" simulation_list.py $targetdir
echo "water_fraction = True" >>$targetdir/config.py
cd "$targetdir"
cd ..
py3clean $targetdir
rm "cli.zip"
zip -r cli.zip ./cli
