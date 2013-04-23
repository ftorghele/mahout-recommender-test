#!/bin/sh

echo "Converting users to csv..."

cat user.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ./users.csv

echo "Converting ratings to csv..."

cat ratings.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ./ratings.csv

