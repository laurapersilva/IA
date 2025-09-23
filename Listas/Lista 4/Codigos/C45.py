import numpy as np
from collections import Counter

class Node:
    def __init__(self, atributo=None, limiar=None, filhos=None, rotulo=None, is_leaf=False):
        self.atributo = atributo        # Índice do atributo (coluna) a dividir
        self.limiar = limiar            # Limiar usado para divisão (atributo contínuo)
        self.filhos = filhos or {}      # Dicionário {condição: nó filho}
        self.rotulo = rotulo            # Classe (para folhas)
        self.is_leaf = is_leaf          # É folha?

    def __str__(self, nivel=0):
        ident = "  " * nivel
        if self.is_leaf:
            return f"{ident}Leaf: {self.rotulo}\n"
        s = f"{ident}[Atributo {self.atributo} < {self.limiar}]\n"
        for cond, filho in self.filhos.items():
            s += f"{ident}  → {cond}:\n{filho.__str__(nivel + 2)}"
        return s

def entropia(y):
    """
    Calcula a entropia de um vetor de classes y.
    """
    contagem = Counter(y)
    total = len(y)
    return -sum((n / total) * np.log2(n / total) for n in contagem.values())

def ganho_informacao(X_col, y, limiar):
    """
    Calcula o ganho de informação para uma divisão binária com um limiar.
    """
    esquerda_y = [y[i] for i in range(len(y)) if X_col[i] < limiar]
    direita_y  = [y[i] for i in range(len(y)) if X_col[i] >= limiar]

    if not esquerda_y or not direita_y:
        return 0, 0  # divisão inválida

    entropia_total = entropia(y)
    ent_esq = entropia(esquerda_y)
    ent_dir = entropia(direita_y)

    p_esq = len(esquerda_y) / len(y)
    p_dir = len(direita_y) / len(y)

    ent_div = p_esq * ent_esq + p_dir * ent_dir
    info_gain = entropia_total - ent_div

    # Split Info (para gain ratio)
    split_info = -sum(p * np.log2(p) for p in [p_esq, p_dir] if p > 0)

    gain_ratio = info_gain / split_info if split_info != 0 else 0

    return info_gain, gain_ratio

def melhor_divisao(X, y):
    """
    Encontra a melhor divisão (atributo e limiar) com maior gain ratio.
    """
    n_atributos = len(X[0])
    melhor_atributo = None
    melhor_limiar = None
    melhor_gain_ratio = -1

    for atributo in range(n_atributos):
        valores = [x[atributo] for x in X]
        valores_unicos = sorted(set(valores))

        # Testa médias entre valores consecutivos
        candidatos = [(valores_unicos[i] + valores_unicos[i + 1]) / 2
                      for i in range(len(valores_unicos) - 1)]

        for limiar in candidatos:
            _, gain_ratio = ganho_informacao(valores, y, limiar)
            if gain_ratio > melhor_gain_ratio:
                melhor_gain_ratio = gain_ratio
                melhor_atributo = atributo
                melhor_limiar = limiar

    return melhor_atributo, melhor_limiar

def construir_arvore(X, y, profundidade_max=None, profundidade=0):
    """
    Constrói a árvore C4.5 recursivamente.
    """
    if len(set(y)) == 1:
        return Node(rotulo=y[0], is_leaf=True)

    if not X or (profundidade_max is not None and profundidade >= profundidade_max):
        # Retorna a classe mais comum
        rotulo_mais_comum = Counter(y).most_common(1)[0][0]
        return Node(rotulo=rotulo_mais_comum, is_leaf=True)

    atributo, limiar = melhor_divisao(X, y)
    if atributo is None:
        rotulo_mais_comum = Counter(y).most_common(1)[0][0]
        return Node(rotulo=rotulo_mais_comum, is_leaf=True)

    # Divide os dados
    esquerda_X = [x for x in X if x[atributo] < limiar]
    esquerda_y = [y[i] for i in range(len(y)) if X[i][atributo] < limiar]

    direita_X = [x for x in X if x[atributo] >= limiar]
    direita_y = [y[i] for i in range(len(y)) if X[i][atributo] >= limiar]

    if not esquerda_y or not direita_y:
        rotulo_mais_comum = Counter(y).most_common(1)[0][0]
        return Node(rotulo=rotulo_mais_comum, is_leaf=True)

    # Recursão
    filho_esq = construir_arvore(esquerda_X, esquerda_y, profundidade_max, profundidade + 1)
    filho_dir = construir_arvore(direita_X, direita_y, profundidade_max, profundidade + 1)

    return Node(
        atributo=atributo,
        limiar=limiar,
        filhos={'<': filho_esq, '>=': filho_dir},
        is_leaf=False
    )

def prever(amostra, arvore):
    if arvore.is_leaf:
        return arvore.rotulo
    valor = amostra[arvore.atributo]
    if valor < arvore.limiar:
        return prever(amostra, arvore.filhos['<'])
    else:
        return prever(amostra, arvore.filhos['>='])

def avaliar(X, y, arvore):
    previsoes = [prever(x, arvore) for x in X]
    acertos = sum(1 for y1, y2 in zip(y, previsoes) if y1 == y2)
    return acertos / len(y)

# Treinar árvore com C4.5
arvore_c45 = construir_arvore(X, y)

# Visualizar a árvore
print(arvore_c45)

# Avaliar
acc = avaliar(X, y, arvore_c45)
print(f"Acurácia no conjunto de treino: {acc:.2f}")