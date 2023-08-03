# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:46:35 2023

@author: Caio Magnani
"""

#Importando
import numpy as np 

#Matrizes
#entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
#saidas = np.array([0,0,0,1])
entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,1,1,1])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1

#Função de ativação
def stepfunction(soma):
    if (soma >= 1):
        return 1 
    return 0

#Multiplicando pelos pesos
def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepfunction(s)

#Encontrando o peso correto
def treinar():
    erroTotal = 1
    while (erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
            print('Peso atualizado ' + str(pesos[j]))
            print('Total de erros: ' + str(erroTotal))
            
treinar()
