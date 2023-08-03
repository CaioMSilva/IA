import numpy as np

#formula do sigmoid
def sigmoid(soma):
    return 1/(1 + np.exp(-soma))

#formula derivada
def sigmoidDerivada(sig):
    return sig * (1-sig)

#entrada saída pesos e epoca
entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]])

saidas = np.array([[0],[1],[1],[0]])

#pesos0 = np.array([[-0.424, -0.740 , -0.961],
                   #[0.358, -0.577, -0.469]])

#pesos1 = np.array([[0.017], [-0.893], [0.148]])

pesos0 = 2*np.random.random((2,3)) - 1
pesos1 = 2*np.random.random((3,1)) - 1

epocas = 1000000
taxadeAprendizagem = 0.7
momento = 1

#camada oculta
for j in range(epocas):
        camadaEntrada = entradas
        somaSinapse0 = np.dot(camadaEntrada, pesos0)
        camadaOculta = sigmoid(somaSinapse0)

#saindo da camada oculta
        somaSinapse1 = np.dot(camadaOculta, pesos1)
        camadaSaida = sigmoid(somaSinapse1)
        
#calculando erro
        erroCamadaSaida = saidas - camadaSaida
        mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
        print("Erro: " +str(mediaAbsoluta))

#calculando delta saida
        derivadaSaida = sigmoidDerivada(camadaSaida)
        deltaSaida = erroCamadaSaida * derivadaSaida

#calculando delta oculto
        pesos1Transposta = pesos1.T 
        deltaSaidaXPeso = deltaSaida.dot(pesos1.T)
        deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

#calculando novos pesos para saída
        camadaOcultaTransposta = camadaOculta.T
        pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
        pesos1 = (pesos1 * momento) + (pesosNovo1 * taxadeAprendizagem)
    
#calculando novos pesos para camada oculta
        camadaEntradaTransposta = camadaEntrada.T
        pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
        pesos0 = (pesos0 * momento) + (pesosNovo0 * taxadeAprendizagem)
        
    