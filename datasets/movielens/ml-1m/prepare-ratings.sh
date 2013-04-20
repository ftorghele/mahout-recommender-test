#!/bin/sh

if [ $# -ne 1 ]
then
  echo -e "\nYou have to download the Movielens 1M dataset from http://www.grouplens.org/node/73 before"
  echo -e "you can run this example. After that extract it and supply the path to the ratings.dat file.\n"
  echo -e "Syntax: $0 /path/to/ratings.dat\n"
  exit -1
fi

RATINGS_COUNT=`wc -l ratings.dat | cut -d ' ' -f 1`
echo "ratings count: $RATINGS_COUNT"
SET_SIZE=`expr $RATINGS_COUNT / 5`
echo "set size: $SET_SIZE"
REMAINDER=`expr $RATINGS_COUNT % 5`
echo "remainder: $REMAINDER"

for i in 1 2 3 4 5
  do
    head -`expr $i \* $SET_SIZE` ratings.dat | tail -$SET_SIZE > r$i.test

    head -`expr \( $i - 1 \) \* $SET_SIZE` ratings.dat > r$i.train
    tail -`expr \( 5 - $i \) \* $SET_SIZE` ratings.dat >> r$i.train

    if [ $i -eq 5 ]; then
       tail -$REMAINDER ratings.dat >> r5.test
    else
       tail -$REMAINDER ratings.dat >> r$i.train
    fi

    echo "r$i.test created.  `wc -l r$i.test | cut -d " " -f 1` lines."
    echo "r$i.train created.  `wc -l r$i.train | cut -d " " -f 1` lines."
done

echo "Converting ratings to csv..."

cat ratings.dat |sed -e s/::/,/g| cut -d, -f1,2,3 > ./ratings.csv

for i in 1 2 3 4 5
  do
    cat r$i.test |sed -e s/::/,/g| cut -d, -f1,2,3 > ./r$i.test.csv
    rm r$i.test
    cat r$i.train |sed -e s/::/,/g| cut -d, -f1,2,3 > ./r$i.train.csv
    rm r$i.train
done
