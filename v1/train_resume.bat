@echo off
chcp 65001 > NUL

python src/train.py --data data/%1 --resume train/%2 ^
%3 %4 %5 %6 %7 %8 %9 
