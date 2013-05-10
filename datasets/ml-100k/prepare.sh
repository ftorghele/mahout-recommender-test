#!/bin/sh

echo "Converting..."

cat ratings.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ./ratings.csv
cat user.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ./users.csv
