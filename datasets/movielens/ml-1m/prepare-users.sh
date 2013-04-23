#!/bin/sh

if [ $# -ne 1 ]
then
  echo -e "\nYou have to download the Movielens 1M dataset from http://www.grouplens.org/node/73 before"
  echo -e "you can run this example. After that extract it and supply the path to the users.dat file.\n"
  echo -e "Syntax: $0 /path/to/users.dat\n"
  exit -1
fi

echo "Converting users to csv..."

cat users.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ./users.csv

