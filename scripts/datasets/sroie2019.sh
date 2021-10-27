#!/bin/bash

start=$(date +%s)

# If you do not provide the directory then by default it is the ~/SROIE20219 directory
if [ -z "$1" ]; then
  # navigate to ~/data
  echo "Navigating to ~/SROIE2019/ ..."
  mkdir -p ~/SROIE2019
  cd ~/SROIE2019/
# check if is valid directory
elif [ ! -d "$1" ]; then
  # shellcheck disable=SC2086
  echo $1 "is not a valid directory"
  exit 0
else
  echo "Navigating to" "$1" "..."
  cd "$1" || exit
fi

echo "Downloading the Scanned receipt dataset..."
curl -L "" -o SROIE2019.zip
echo "Done downloading."

# Extract data
echo "Extracting the dataset..."
unzip SROIE2019.zip
echo "removing the zip file..."
rm SROIE2019.zip

end=$(date +%s)
runtime=$((end - start))

echo "Completed in" $runtime "seconds"
