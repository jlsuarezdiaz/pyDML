#!/bin/bash
#Argumentos:
# $1- Nombre del fichero lex
# $2- Nombre del programa

flex $1
g++ -g -o $2 lex.yy.c -lfl -ly