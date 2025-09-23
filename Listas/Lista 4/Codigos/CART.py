import numpy as np
from collections import Counter

class CARTNode:
    def __init__(self, atributo=None, limiar=None, filhos=None, rotulo=None, is_leaf=False):
        self.atributo = atributo        # Índice do atributo usado para dividir
        self.limiar = limiar            # Limiar para atributos contínuos
        self.filhos = filhos or {}      # {'<': filho_esq, '>=': filho_dir}
        self.rotulo = rotulo            # Classe dominante (se folha)
        self.is_leaf = is_leaf          # É folha?

    def __str__(self, nivel=0):
        ident = "  " * nivel
        if self.is_leaf:
            return f"{ident}Leaf: {self.rotulo}\n"
        s = f"{ident}[X{self.atributo} < {self.limiar}]\n"
        for cond, filho in self.filhos.items():
            s += f"{ident}  → {cond}:\n{filho.__str__(nivel + 2)}"
        return s

def gini_impureza(y):
    """
    Calcula a impureza de Gini para um vetor de rótulos y.
    """
    total = len(y)
    if total == 0:
        return 0
    contagem = Counter(y)
    return 1 - sum((n / total) ** 2 for n in contagem.values())

def melhor_divisao_CART(X, y):
    """
    Encontra a melhor divisão (atributo + limiar) que minimiza a impureza de Gini.
    """
    n_atributos = len(X[0])
    melhor_atributo = None
    melhor_limiar = None
    melhor_gini = float('inf')

    for atributo in range(n_atributos):
        valores = [x[atributo] for x in X]
        valores_unicos = sorted(set(valores))
        
        # Tenta limiares entre valores consecutivos
        candidatos = [(valores_unicos[i] + valores_unicos[i + 1]) / 2
                      for i in range(len(valores_unicos) - 1)]

        for limiar in candidatos:
            esquerda_y = [y[i] for i in range(len(y)) if X[i][atributo] < limiar]
            direita_y  = [y[i] for i in range(len(y)) if X[i][atributo] >= limiar]

            if not esquerda_y or not direita_y:
                continue  # divisão inválida

            gini_esq = gini_impureza(esquerda_y)
            gini_dir = gini_impureza(direita_y)

            p_esq = len(esquerda_y) / len(y)
            p_dir = len(direita_y) / len(y)

            gini_total = p_esq * gini_esq + p_dir * gini_dir

            if gini_total < melhor_gini:
                melhor_gini = gini_total
                melhor_atributo = atributo
                melhor_limiar = limiar

    return melhor_atributo, melhor_limiar

def construir_arvore_CART(X, y, profundidade_max=None, profundidade=0):
    """
    Constrói a árvore CART recursivamente.
    """
    if len(set(y)) == 1:
        return CARTNode(rotulo=y[0], is_leaf=True)

    if not X or (profundidade_max is not None and profundidade >= profundidade_max):
        rotulo_mais_comum = Counter(y).most_common(1)[0][0]
        return CARTNode(rotulo=rotulo_mais_comum, is_leaf=True)

    atributo, limiar = melhor_divisao_CART(X, y)
    if atributo is None:
        rotulo_mais_comum = Counter(y).most_common(1)[0][0]
        return CARTNode(rotulo=rotulo_mais_comum, is_leaf=True)

    esquerda_X = [x for x in X if x[atributo] < limiar]
    esquerda_y = [y[i] for i in range(len(y)) if X[i][atributo] < limiar]

    direita_X = [x for x in X if x[atributo] >= limiar]
    direita_y = [y[i] for i in range(len(y)) if X[i][atributo] >= limiar]

    if not esquerda_y or not direita_y:
        rotulo_mais_comum = Counter(y).most_common(1)[0][0]
        return CARTNode(rotulo=rotulo_mais_comum, is_leaf=True)

    filho_esq = construir_arvore_CART(esquerda_X, esquerda_y, profundidade_max, profundidade + 1)
    filho_dir = construir_arvore_CART(direita_X, direita_y, profundidade_max, profundidade + 1)

    return CARTNode(
        atributo=atributo,
        limiar=limiar,
        filhos={'<': filho_esq, '>=': filho_dir},
        is_leaf=False
    )

def prever_CART(amostra, arvore):
    if arvore.is_leaf:
        return arvore.rotulo
    valor = amostra[arvore.atributo]
    if valor < arvore.limiar:
        return prever_CART(amostra, arvore.filhos['<'])
    else:
        return prever_CART(amostra, arvore.filhos['>='])

def avaliar_CART(X, y, arvore):
    previsoes = [prever_CART(x, arvore) for x in X]
    acertos = sum(1 for y1, y2 in zip(y, previsoes) if y1 == y2)
    return acertos / len(y)

rvore_cart = construir_arvore_CART(X, y)

# Exibindo a árvore
print(arvore_cart)

# Avaliando no mesmo conjunto
acc = avaliar_CART(X, y, arvore_cart)
print(f"Acurácia no conjunto de treino: {acc:.2f}")