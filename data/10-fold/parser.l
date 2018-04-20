                 /*----- Sección de Declaraciones --------------*/ 
%{
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

/*
 * Número de medidas.
 * En esta variable se debe anotar el número de medidas que vamos a tomar.
 */
 static const int NUM_MEDIDAS = 1;

/*
 * Número de filas y columnas que tendrá el fichero csv que vamos a crear.
 */
 static const int NUM_FILAS = 3;
 static const int NUM_COLUMNAS = 12;


/* Código interno (no modificar) */
 ofstream *outputs[NUM_MEDIDAS];
 static int nextMedida = 0;

 static int filas = 0;
 static int columnas = 0;

 void end();

 #define WRITE(i)		if(nextMedida==i){\
		*(outputs[i]) << "\""<< yytext << "\",";\
		 nextMedida++; \
	} \
	else{\
			REJECT;\
	}\


/* EXPLICACIÓN DE LAS SECCIONES QUE SE MUESTRAN EN LOS SIGUIENTES BLOQUES: */

/* Sección de cabeceras:
 * Cada cabecera representa un alias con lo que precede a la medida
 * correspondiente que queramos tomar.
 * El texto restante tras la última medida debe anotarse en la siguiente
 * cabecera vacía.
 */


/* ... [Añadir más si es necesario.] 
 * Sintaxis: cab$i <tabulador> (Expresión regular) */

/* Sección de medidas:
 * Cada expresión regular de esta sección debe representar
 * la estructura que tiene la medida que se va a tomar.
 */

 /* ... [Añadir más si es necesario.] 
  * Sintaxis: med$i <tabulador> (Expresión regular) */
 
/* Reglas para las cabeceras:
 * Las cabeceras no imprimen información.
 */

/* ... [Añadir más cabeceras a la expresión regular si es necesario] */
/* Reglas para las medidas:
 * Las medidas imprimen la información en el fichero csv correspondiente.
 */

/*{med1}  {if(nextMedida==1){ outputs[1] << yytext << ","; nextMedida++; } else REJECT;} */
/*{med2}  {if(nextMedida==2){ outputs[2] << yytext << ","; nextMedida++; } else REJECT;} */
/*{med3}  {if(nextMedida==3){ outputs[3] << yytext << ","; nextMedida++; } else REJECT;} */
/*{med4}  {if(nextMedida==4){ outputs[4] << yytext << ","; nextMedida++; } else REJECT;} */

/* ... [Añadir más si es necesario] 
 * Sintaxis: {med$i}  {if(nextMedida==$i){ outputs[$i] << yytext; nextMedida++; } else REJECT;}
 */

%}





letra   [a-zA-Z]
digito  [0-9]
entero  {digito}+
real    {entero}\.{entero}
real2   {real}[eE][\+\-]?{entero}
numero  (\+|-)?{entero}|{real}|{real2}

text_real real\[{numero},{numero}]
text_integer integer\[{numero},{numero}]
input  @inputs(.*\n)
output @outputs(.*\n)
class  \{.*\}

space		[ ]+
tab			\t+
white		[ \t]+
lines		\n+
anywhite	[ \t\n]+

cab0		("\nTAMAÑO DE MUESTRA "{numero}{lines})?"\nNÚMERO DE THREADS = "{numero}{lines}"CÁLCULO DE PI\nValor de PI (calculado secuencialmente)  = "{numero}"\tTiempo: "{numero}"\nValor de PI (calculado concurrentemente) = "{numero}"\tTiempo: "
cabEnd	"\n"
med0		{numero}




%%
                 /*----- Sección de Reglas ----------------*/ 

{text_real}       {printf("numeric");}  
{text_integer}    {printf("numeric");}
{input}           {printf("\n");}
{output}          {printf("\n");}
{class}           {printf(" %s",yytext);}

.				{printf("%s",yytext);}

%%   
                /*----- Sección de Procedimientos --------*/ 




void end(){
	nextMedida = 0;
	columnas = (columnas + 1) % NUM_COLUMNAS;
	if(columnas==0){
		filas = (filas + 1) % NUM_FILAS;
		for(int i = 0; i < NUM_MEDIDAS; i++){
			*(outputs[i]) << "\n";
		}
	}


}

int main (int argc, char *argv[])
{   

  for(int i = 1; i < argc; i++){
        yyin = fopen (argv[i], "rt"); 
        yylex();  
  }  
  
  return 0; 
}
