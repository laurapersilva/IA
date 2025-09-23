import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Função para calcular o ganho de informação
def calcular_ganho(entropia_inicial, entropia_final):
    return entropia_inicial - entropia_final

# Função para calcular a entropia de um conjunto de dados
def calcular_entropia(dados):
    """
    Calcula a entropia de um conjunto de dados (discretizados ou contínuos).
    A entropia mede a incerteza ou aleatoriedade de um conjunto de dados.
    """
    valores, contagens = np.unique(dados, return_counts=True)
    probabilidade = contagens / len(dados)
    entropia = -np.sum(probabilidade * np.log2(probabilidade))
    return entropia

# Função para tratar atributos não nominais (por exemplo, valores contínuos)
def tratar_atributos_nao_nominais(dados, limiar=0.5):
    """
    Discretiza atributos contínuos em dois grupos: acima e abaixo do limiar.
    """
    return np.array([1 if valor >= limiar else 0 for valor in dados])

# Função para gerar novos estados com base em transições (para simplificação)
def gerar_novos_estados(estados):
    """
    Gera novos estados perturbando os estados atuais com um pouco de ruído.
    """
    return estados + np.random.randn(*estados.shape) * 0.1  # Perturba os estados com ruído

# Função para calcular um interpolante entre dois estados (simplificado)
def calcular_interpolante(estado1, estado2):
    """
    Calcula um interpolante simples entre dois estados, retornando sua média.
    """
    return (estado1 + estado2) / 2  # Exemplo simples: média entre os estados

# Função principal para o algoritmo IC3 com árvores
def IC3(M, P, I, num_itens=5):
    """
    M: Modelo do sistema
    P: Propriedade a ser verificada
    I: Conjunto de estados iniciais
    num_itens: Quantidade de estados a serem gerados em cada iteração
    """
    
    # Inicializa a árvore IC3 e os estados
    estados = np.array(I)  # Definindo o estado inicial
    árvore = []

    # 1. Propagação de estados
    for it in range(num_itens):
        # Calcula a entropia do conjunto atual de estados
        entropia_inicial = calcular_entropia(estados)
        
        # Gerar novos estados através de transições (exemplo simples)
        novos_estados = gerar_novos_estados(estados)

        # Tratar atributos não nominais (se existirem)
        novos_estados_discretizados = tratar_atributos_nao_nominais(novos_estados)

        # Calcula a entropia dos novos estados
        entropia_final = calcular_entropia(novos_estados_discretizados)

        # Cálculo do ganho de informação
        ganho = calcular_ganho(entropia_inicial, entropia_final)

        # Verificação da propriedade (simulando, pode ser algo como verificar se a propriedade é violada)
        propriedade_satisfeita = P(estados)
        
        if not propriedade_satisfeita:
            # Se a propriedade não for satisfeita, calculamos um interpolante
            interpolante = calcular_interpolante(estados, novos_estados)
            árvore.append(interpolante)  # Adiciona o interpolante à árvore IC3

        estados = novos_estados  # Atualiza os estados para a próxima iteração

    return árvore

# Exemplo de modelo e propriedade:
def modelo_exemplo(estados):
    """
    Função de exemplo para simular a propriedade a ser verificada.
    A propriedade é violada se algum estado tiver um valor maior que 10 (apenas ilustrativo)
    """
    return np.all(estados <= 10)

# Conjunto de estados iniciais (simulando valores contínuos)
estados_iniciais = np.random.randn(10, 2) * 5  # 10 estados, cada um com 2 atributos

# Chamando o algoritmo IC3
arvore = IC3(None, modelo_exemplo, estados_iniciais, num_itens=10)

print("Árvore gerada:", arvore)

# ============================
# Usando DecisionTreeClassifier como baseline (apenas para verificação)
# ============================

# 1. Gerando rótulos para os dados, usando a mesma propriedade
rotulos = np.array([1 if modelo_exemplo(estado) else 0 for estado in estados_iniciais])

# 2. Treinando o DecisionTreeClassifier (apenas como baseline)
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(estados_iniciais, rotulos)

# 3. Fazendo previsões com o modelo treinado
predicoes = clf.predict(estados_iniciais)

# 4. Calculando a acurácia
acuracia = accuracy_score(rotulos, predicoes)

print(f"Acurácia do DecisionTreeClassifier: {acuracia:.2f}")
