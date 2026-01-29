!pip install pyspark

from pyspark.sql import SparkSession

spark = SparkSession.builder\
  .master('local[*]')\
  .appName('Regressao Spark')\
  .getOrCreate()

spark
#vai rodar local, usando totods os nucleos da cpu (VM do google)

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
