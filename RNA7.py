import math
#XOR
Dados = [
	[0.0, 0.0],
	[1.0, 0.0],
	[0.0, 1.0],
	[1.0, 1.0]
]
Classe = [0.0, 1.0, 1.0, 0.0]
PesosCamadaEntrada = [0.1,0.2,0.3,0.4,0.1,0.3,0.0,0.1]
PesosCamadaIntermediara = [0.1,0.1,0.2,0.0,0.0,0.2,0.1,0.2]
PesosCamadaSaida = [0.0,0.0]
Bias = 1.0
PesosBias = [0.0, 0.2, 0.3,0.1,0.2,0.3,0.0]
TaxaDeAprentisado = 0.4
Epocas = 0
ErroDaRede = 0.0
ErroQuadrado = 0.0
aux = 0
Parar = 0
def sigmoid(calculo):
  return 1 / (1 + math.exp(-calculo))

while Epocas <= 20000 and Parar==0:
	print("Treinamento Iniciado Tempo: ",Epocas)
	for percore in range(4):
		eNA = Dados[percore][0]*PesosCamadaEntrada[0] + Dados[percore][1]*PesosCamadaEntrada[1] + PesosBias[0]*Bias
		eNA = sigmoid(eNA)
		eNB = Dados[percore][0]*PesosCamadaEntrada[2] + Dados[percore][1]*PesosCamadaEntrada[3] + PesosBias[1]*Bias
		eNB = sigmoid(eNB)
		eNC = Dados[percore][0]*PesosCamadaEntrada[4] + Dados[percore][1]*PesosCamadaEntrada[5] + PesosBias[2]*Bias
		eNC = sigmoid(eNC)
		eND = Dados[percore][0]*PesosCamadaEntrada[6] + Dados[percore][1]*PesosCamadaEntrada[7] + PesosBias[3]*Bias
		eND = sigmoid(eND)
		iNA = eNA*PesosCamadaIntermediara[0] + eNB*PesosCamadaIntermediara[1] + eNC*PesosCamadaIntermediara[2] + eND*PesosCamadaIntermediara[3] + PesosBias[4]*Bias
		iNA = sigmoid(iNA)
		iNB = eNA*PesosCamadaIntermediara[4] + eNB*PesosCamadaIntermediara[5] + eNC*PesosCamadaIntermediara[6] + eND*PesosCamadaIntermediara[7] + PesosBias[5]*Bias
		iNB = sigmoid(iNB)
		sNA = iNA*PesosCamadaSaida[0] + iNB*PesosCamadaSaida[1] + PesosBias[6]*Bias
		sNA = sigmoid(sNA)
		ErroDaRede = Classe[percore] - sNA
		print("Erro total da rede: ",ErroDaRede)
		#ajustes
		Erro_saida = ErroDaRede * sNA * (1 - sNA)
		Erro_iNA = iNA * (1 - iNA) * PesosCamadaSaida[0] * Erro_saida
		Erro_iNB = iNB * (1 - iNB) * PesosCamadaSaida[1] * Erro_saida
		Erro_eNA = eNA * (1 - eNA) * (PesosCamadaIntermediara[0]*Erro_saida + PesosCamadaIntermediara[1]*Erro_saida)
		Erro_eNB = eNB * (1 - eNB) * (PesosCamadaIntermediara[2]*Erro_saida + PesosCamadaIntermediara[3]*Erro_saida)
		Erro_eNC = eNC * (1 - eNC) * (PesosCamadaIntermediara[4]*Erro_saida + PesosCamadaIntermediara[5]*Erro_saida)
		Erro_eND = eND * (1 - eND) * (PesosCamadaIntermediara[6]*Erro_saida + PesosCamadaIntermediara[7]*Erro_saida)
		
		#ajuste de pesos saida
		PesosBias[6] = PesosBias[6] + TaxaDeAprentisado * Erro_saida * Bias
		PesosCamadaSaida[0] = PesosCamadaSaida[0] + Erro_saida * TaxaDeAprentisado * iNA
		PesosCamadaSaida[1] = PesosCamadaSaida[1] + Erro_saida * TaxaDeAprentisado * iNB
		
		#ajuste de pesos Intermetiaria
		PesosCamadaIntermediara[0] = PesosCamadaIntermediara[0] + Erro_iNA * TaxaDeAprentisado * eNA
		PesosCamadaIntermediara[1] = PesosCamadaIntermediara[1] + Erro_iNA * TaxaDeAprentisado * eNB
		PesosCamadaIntermediara[2] = PesosCamadaIntermediara[2] + Erro_iNA * TaxaDeAprentisado * eNC
		PesosCamadaIntermediara[3] = PesosCamadaIntermediara[3] + Erro_iNA * TaxaDeAprentisado * eND
		PesosBias[4] = PesosBias[4] + TaxaDeAprentisado * Erro_iNA * Bias
		
		PesosBias[5] = PesosBias[5] + TaxaDeAprentisado * Erro_iNB * Bias
		PesosCamadaIntermediara[4] = PesosCamadaIntermediara[4] + Erro_iNB * TaxaDeAprentisado * eNA
		PesosCamadaIntermediara[5] = PesosCamadaIntermediara[5] + Erro_iNB * TaxaDeAprentisado * eNB
		PesosCamadaIntermediara[6] = PesosCamadaIntermediara[6] + Erro_iNB * TaxaDeAprentisado * eNC
		PesosCamadaIntermediara[7] = PesosCamadaIntermediara[7] + Erro_iNB * TaxaDeAprentisado * eND
		
		#ajuste Entrada
		PesosBias[0] = PesosBias[0] + TaxaDeAprentisado * Erro_eNA * Bias
		PesosCamadaEntrada[0] = PesosCamadaEntrada[0] + TaxaDeAprentisado * Erro_eNA * Dados[percore][0]
		PesosCamadaEntrada[1] = PesosCamadaEntrada[1] + TaxaDeAprentisado * Erro_eNA * Dados[percore][1]
		
		PesosBias[1] = PesosBias[1] + TaxaDeAprentisado * Erro_eNB * Bias
		PesosCamadaEntrada[2] = PesosCamadaEntrada[2] + TaxaDeAprentisado * Erro_eNB * Dados[percore][0]
		PesosCamadaEntrada[3] = PesosCamadaEntrada[3] + TaxaDeAprentisado * Erro_eNB * Dados[percore][1]
		
		PesosBias[2] = PesosBias[2] + TaxaDeAprentisado * Erro_eNC * Bias
		PesosCamadaEntrada[4] = PesosCamadaEntrada[4] + TaxaDeAprentisado * Erro_eNC * Dados[percore][0]
		PesosCamadaEntrada[5] = PesosCamadaEntrada[5] + TaxaDeAprentisado * Erro_eNC * Dados[percore][1]
		
		PesosBias[3] = PesosBias[3] + TaxaDeAprentisado * Erro_eND * Bias
		PesosCamadaEntrada[6] = PesosCamadaEntrada[6] + TaxaDeAprentisado * Erro_eND * Dados[percore][0]
		PesosCamadaEntrada[7] = PesosCamadaEntrada[7] + TaxaDeAprentisado * Erro_eND * Dados[percore][1]
		
		aux +=1
		if(aux >3):
			aux=0
			ErroQuadrado = ErroQuadrado + (ErroDaRede*ErroDaRede)
			if(ErroQuadrado <= 0.001):
				Parar=1
				print("Convergiu!",ErroQuadrado)
			else:
				ErroQuadrado=0
				Parar=0
		else:
			ErroQuadrado = ErroQuadrado + (ErroDaRede*ErroDaRede)
	
	Epocas += 1

print("#"*50)

def testar():
    x = float(input("Digite o valor de x: "))
    y = float(input("Digite o valor de y: "))
    e1 = x*PesosCamadaEntrada[0] + y*PesosCamadaEntrada[1] + PesosBias[0]*Bias
    e1 = sigmoid(e1)
    e2 = x*PesosCamadaEntrada[2] + y*PesosCamadaEntrada[3] + PesosBias[1]*Bias
    e2 = sigmoid(e2)
    e3 = x*PesosCamadaEntrada[4] + y*PesosCamadaEntrada[5] + PesosBias[2]*Bias
    e3 = sigmoid(e3)
    e4 = x*PesosCamadaEntrada[6] + y*PesosCamadaEntrada[7] + PesosBias[3]*Bias
    e4 = sigmoid(e4)
    i1 = e1*PesosCamadaIntermediara[0]+e2*PesosCamadaIntermediara[1]+e3*PesosCamadaIntermediara[2]+e4*PesosCamadaIntermediara[3]+PesosBias[4]*Bias 
    i1 = sigmoid(i1)
    i2 = e1*PesosCamadaIntermediara[4]+e2*PesosCamadaIntermediara[5]+e3*PesosCamadaIntermediara[6]+e4*PesosCamadaIntermediara[7]+PesosBias[5]*Bias 
    i2 = sigmoid(i2)
    s1 = i1*PesosCamadaSaida[0] + i2*PesosCamadaSaida[1] + PesosBias[6]*Bias
    s1 = sigmoid(s1)
    if(s1 <= 0.5):
	    media = 0
    else:
	    media=1
    print(s1)	
    print("Valor: ",x," ",y," = ",media)
    usuario = int(input("Digite 1 para sair"))
    if(usuario != 1):
    	testar()
    else:
    	exit()

testar()    					
