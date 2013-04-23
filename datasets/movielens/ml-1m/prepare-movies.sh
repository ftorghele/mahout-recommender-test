#!/bin/sh

if [ $# -ne 1 ]
then
  echo -e "\nYou have to download the Movielens 1M dataset from http://www.grouplens.org/node/73 before"
  echo -e "you can run this example. After that extract it and supply the path to the movies.dat file.\n"
  echo -e "Syntax: $0 /path/to/movies.dat\n"
  exit -1
fi

echo "Converting movies to csv..."

cat movies.dat |sed -e s/,/./g| sed -e s/::/,/g| cut -d, -f1,2,3 > ./movies.csv

