!pip install pyspark

from pyspark.sql import SparkSession

spark = SparkSession.builder\
  .master('local[*]')\
  .appName('Regressao Spark')\
  .getOrCreate()

spark
#vai rodar local, usando todos os nucleos da cpu (VM do google)

from google.colab import drive
drive.mount('/content/drive')

dados = spark.read.json(
    '/content/drive/MyDrive/SparkRegressao/imoveis.json'
)

dados\
  .show(truncate = False)

dados.count()

dados.printSchema()

dataset = dados\
  .select('ident.customerID', 'listing.types.*', 'listing.features.*', 'listing.address.*', 'listing.prices.price', 'listing.prices.tax.*')\
  .drop('city', 'location', 'totalAreas')

dataset.show(truncate = False)

dataset.printSchema()

from pyspark.sql.types import IntegerType, DoubleType

dataset = dataset\
  .withColumn('usableAreas', dataset['usableAreas'].cast(IntegerType()))\
  .withColumn('price', dataset['price'].cast(DoubleType()))\
  .withColumn('condo', dataset['condo'].cast(DoubleType()))\
  .withColumn('iptu', dataset['iptu'].cast(DoubleType()))

dataset.show()

dataset\
  .select('usage')\
  .groupBy('usage')\
  .count()\
  .show()

dataset = dataset\
            .select('*')\
            .where('usage == "Residencial"')

dataset\
  .select('unit')\
  .groupBy('unit')\
  .count()\
  .show()

dataset\
  .select('zone')\
  .groupBy('zone')\
  .count()\
  .show()

from pyspark.sql import functions as f

dataset.show()

dataset\
    .select([f.count(f.when(f.isnull(c), True)).alias(c) for c in dataset.columns])\
    .show()
#tem que usar o alias para manter o nome original da coluna
#isnull(c) é linha por linha de cada coluna

dataset = dataset\
  .select('*')\
  .na\
  .fill(0)\

#na\ seleciona apenas os parametros nulos
#.fill(0) preenche os valores NULL com 0

dataset\
    .select([f.count(f.when(f.isnull(c), True)).alias(c) for c in dataset.columns])\
    .show()

dataset\
  .select('zone')\
  .groupBy('zone')\
  .count()\
  .show()

dataset = dataset\
  .select('*')\
  .where(f.col('zone') != '')

dataset.show()

dataset\
  .groupBy('customerID')\
  .pivot('unit')\
  .agg(f.lit(1))\
  .na\
  .fill(0)\
  .show()
#.groupBy('customerID')\ para cada customerID, faça o pivot
#cada valor distinto em unit, vira uma nova coluna (one-hot-encoding)
#.agg(f.lit(1))\ preenche com 1 quando tiver uma ocorrencia

unit = dataset\
  .groupBy('customerID')\
  .pivot('unit')\
  .agg(f.lit(1))\
  .na\
  .fill(0)

zone = dataset\
  .groupBy('customerID')\
  .pivot('zone')\
  .agg(f.lit(1))\
  .na\
  .fill(0)

dataset = dataset\
  .join(unit, 'customerID', how = 'inner')\
  .join(zone, 'customerID', how = 'inner')
#inner é interseção, so ficam nos resultados os customerID que estao presentes nos tres dataset

dataset.show()

from pyspark.ml.feature import VectorAssembler

dataset = dataset.withColumnRenamed('price', 'label') #spark trabalha com nome padrao de coluna de previsao

x = [
    'bathrooms',
    'bedrooms',
    'floors',
    'parkingSpaces',
    'suites',
    'unitFloor',
    'unitsOnTheFloor',
    'usableAreas',
    'condo',
    'iptu',
    'Apartamento',
    'Casa',
    'Outros',
    'Zona Central',
    'Zona Norte',
    'Zona Oeste',
    'Zona Sul'
]

assembler = VectorAssembler(inputCols = x, outputCol = 'features')

dataset.show()

"""Vetorizando o dataset"""

dataset_prep = assembler.transform(dataset).select('features', 'label')
#assembler.transform(dataset) apenas adiciona uma coluna chamada features
#.select('features', 'label') seleciona as colunas features e label

dataset_prep.show(10, truncate = False)
#(size, [índices], [valores])

"""Análise de Correlação"""

from pyspark.ml.stat import Correlation

correlacao = Correlation.corr(dataset_prep, 'features').collect()
#corr considera cada posicao do vetor como uma variavel
#collect faz o calculo de fato
#retorna matrizes densa, onde a celula [0][0] é a correlação de pearson na forma densa

correlacao

correlacao = correlacao[0][0] #[0][0] é a correlação de pearson

correlacao

import pandas as pd

correlacao.toArray()

dataframe_correlacao = pd.DataFrame(correlacao.toArray(), columns = x, index = x)
#columns = x nome das colunas
#index = x nome das linhas

dataframe_correlacao

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (12, 10))
paleta = sns.color_palette("light:salmon", as_cmap = True)
#as_cmap = True mapa continuos de cores ao inves de diversas cores, cria um gradiente suave
sns.heatmap(dataframe_correlacao.round(1), annot = True, cmap = paleta)
#annot = True escreve o valor dentro da celula
#cmap = paleta define como os valores viram cores
#seaborn calcula cada cor referente a cada numero
