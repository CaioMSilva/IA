from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConection

rede =  FeedForwardNetwork()

camadaEntrada = LinearLayer(2)
camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias1 = BiasUnit()   
bias2 = BiasUnit()  

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

entradaOculta = FullConection(camadaEntrada, camadaOculta)
ocultaSaida = FullConection(camadaOculta, camadaSaida)
biasOculta = FullConection(bias1, camadaOculta)
biasSaida = FullConection(bias2, camadaSaida)

rede.sortModules()

print (rede)