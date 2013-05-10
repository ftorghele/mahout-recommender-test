#!/bin/sh

echo "Converting..."

cat users.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ./users.csv
cat ratings.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ./ratings.csv

