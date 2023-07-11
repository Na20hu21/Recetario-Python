from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('recetas.csv', 'r', encoding='utf-8') as file:
    data = [line.strip().split(', ') for line in file]

recetas = [item[0] for item in data]

vectorizer = CountVectorizer().fit_transform(recetas)
matriz_caracteristicas = vectorizer.toarray()

similitudes = cosine_similarity(matriz_caracteristicas)

def obtener_insumos(receta):
    insumos = []
    for item in data:
        if item[0] == receta:
            insumos = item[1:]
            break
    return insumos

print(obtener_insumos('Milanesa del palacio'))
print(obtener_insumos('Tarta de Frutilla'))
print(obtener_insumos('Cafe con leche'))
