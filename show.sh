#!/bin/bash

for name in `ls | grep py$`
do
	cat $name
done
