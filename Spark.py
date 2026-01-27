# instalar as dependências
!apt-get update -qq
!apt-get install openjdk-8-jdk-headless -qq > /dev/null
!wget -q https://archive.apache.org/dist/spark/spark-3.5.4/spark-3.5.4-bin-hadoop3.tgz
!tar xf spark-3.5.4-bin-hadoop3.tgz
!pip install -q findspark
!pip install pyspark==3.5.4

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.5.4-bin-hadoop3"

import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local[*]') \
    .appName("Iniciando Spark") \
    .config('spark.ui.port', '4050') \
    .getOrCreate()
#pyspark.sql modulo de spark sql
#SparkSession classe principal do spark
#SparkSession.builder definie todas as configuracoes do spark antes de iniciar
#.master('local[*]') executa na prorpia maquina, [*] indica para usar todas as cpus disponiveis
#.appName("Iniciando Spark") nome da aplicação
#.getOrCreate() cria o objeto spark ou reutiliza um existente

#!wget -q https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
#!unzip ngrok-stable-linux-amd64.zip

#get_ipython().system_raw('./ngrok config add-authtoken 38UR94lloU9Y057f988urX7j1SJ_5Ek535BGB9fnyY8yrtbW2')
#get_ipython().system_raw('./ngrok http 4050 &')

#!curl -s http://localhost:4040/api/tunnels

from google.colab import drive
drive.mount('/content/drive')

data = [('Zeca','35'), ('Eva', '29')]
colNames = ['Nome', 'Idade']
df = spark.createDataFrame(data, colNames)
df.show()

df.toPandas()

from google.colab import drive #importa o modulo drive
drive.mount('/content/drive') #permite que o colab leia e escreva diretamente no drive

import zipfile
zipfile.ZipFile('/content/drive/MyDrive/curso-spark/empresas.zip', 'r').extractall('/content/drive/MyDrive/curso-spark')

path = '/content/drive/MyDrive/curso-spark/empresas'

empresas = spark.read.csv(path, sep = ';', inferSchema = True)
#inferSchema = True tenta descobrir o tipo de cada coluna

empresas.count()

zipfile.ZipFile('/content/drive/MyDrive/curso-spark/socios.zip', 'r').extractall('/content/drive/MyDrive/curso-spark/')

path = '/content/drive/MyDrive/curso-spark/socios'
socios = spark.read.csv(path, sep = ';', inferSchema = True)
socios.count()

zipfile.ZipFile('/content/drive/MyDrive/curso-spark/estabelecimentos.zip', 'r').extractall('/content/drive/MyDrive/curso-spark/')

path = '/content/drive/MyDrive/curso-spark/estabelecimentos'
estabelecimentos = spark.read.csv(path, sep = ';', inferSchema = True)
estabelecimentos.count()

empresas.limit(5).toPandas()

empresasColNames = ['cnpj_basico', 'razao_social_nome_empresarial', 'natureza_juridica', 'qualificacao_do_responsavel', 'capital_social_da_empresa', 'porte_da_empresa', 'ente_federativo_responsavel']

for index, colName in enumerate(empresasColNames):
  empresas = empresas.withColumnRenamed(f"_c{index}", colName)
#enumerate serve para percorrer uma lista e ao mesmo tempo obter o indice dos elementos

empresas.columns

estabsColNames = ['cnpj_basico', 'cnpj_ordem', 'cnpj_dv', 'identificador_matriz_filial', 'nome_fantasia', 'situacao_cadastral', 'data_situacao_cadastral', 'motivo_situacao_cadastral', 'nome_da_cidade_no_exterior', 'pais', 'data_de_inicio_atividade', 'cnae_fiscal_principal', 'cnae_fiscal_secundaria', 'tipo_de_logradouro', 'logradouro', 'numero', 'complemento', 'bairro', 'cep', 'uf', 'municipio', 'ddd_1', 'telefone_1', 'ddd_2', 'telefone_2', 'ddd_do_fax', 'fax', 'correio_eletronico', 'situacao_especial', 'data_da_situacao_especial']

sociosColNames = ['cnpj_basico', 'identificador_de_socio', 'nome_do_socio_ou_razao_social', 'cnpj_ou_cpf_do_socio', 'qualificacao_do_socio', 'data_de_entrada_sociedade', 'pais', 'representante_legal', 'nome_do_representante', 'qualificacao_do_representante_legal', 'faixa_etaria']

estabelecimentos.limit(5).toPandas()

for index, nomeCol in enumerate(estabsColNames):
  estabelecimentos = estabelecimentos.withColumnRenamed(f"_c{index}", nomeCol)

estabelecimentos.limit(5).toPandas()

socios.limit(5).toPandas()

for index, nomeCol in enumerate(sociosColNames):
  socios = socios.withColumnRenamed(f"_c{index}", nomeCol)
socios.limit(5).toPandas()

from pyspark.sql.types import DoubleType, StringType
from pyspark.sql import functions as f #modeulo que tem a funcao to_date por exemplo

empresas.printSchema() #mostra o tipo de cada variavel

empresas.limit(5).toPandas()

empresas = empresas.withColumn('capital_social_da_empresa', f.regexp_replace('capital_social_da_empresa', ',', '.'))
#withColumn cria uma nova coluna ou sobrescreve uma existente
#'capital_social_da_empresa' nome da coluna que sera criada ou sobrescrita
empresas.limit(5).toPandas()

empresas = empresas.withColumn('capital_social_da_empresa', empresas['capital_social_da_empresa'].cast(DoubleType()))
#converte o tipo da variavel para doublee
empresas.printSchema()

df = spark.createDataFrame([(20200924,), (20201022,), (20210215,)], ['data']) #spark precisa de tupla
df.toPandas()

df = df.withColumn('data', f.to_date(df.data.cast(StringType()), 'yyyyMMdd'))
df.printSchema()

estabelecimentos.printSchema()

estabelecimentos = estabelecimentos\
  .withColumn('data_de_inicio_atividade', f.to_date(estabelecimentos.data_de_inicio_atividade.cast(StringType()), 'yyyyMMdd'))\
  .withColumn('data_situacao_cadastral', f.to_date(estabelecimentos.data_situacao_cadastral.cast(StringType()), 'yyyyMMdd'))\
  .withColumn('data_da_situacao_especial', f.to_date(estabelecimentos.data_da_situacao_especial.cast(StringType()), 'yyyyMMdd'))

estabelecimentos.printSchema()

empresas\
  .select(['natureza_juridica', 'porte_da_empresa', 'capital_social_da_empresa'])\
  .show(5, False)

socios.printSchema()

socios = socios\
  .withColumn('data_de_entrada_sociedade', f.to_date(socios.data_de_entrada_sociedade.cast(StringType()), 'yyyyMMdd'))

socios.printSchema()

socios\
  .select(['nome_do_socio_ou_razao_social', 'faixa_etaria', f.year('data_de_entrada_sociedade').alias('ano_de_entrada')])\
  .show(5, False)

estabelecimentos\
  .select('nome_fantasia','municipio', f.year('data_de_inicio_atividade').alias('ano_de_inicio_atividade'),f.month('data_de_inicio_atividade').alias('mes_de_inicio_atividade'))\
  .show(5, False)

df  = spark.createDataFrame([(1,), (2,), (3,), (None,)], ['data'])
df.toPandas()

df.show(5)

df  = spark.createDataFrame([(1.,), (2.,), (3.,), (float('nan'),)], ['data'])
df.toPandas()

df.show(5)

df  = spark.createDataFrame([('1',), ('2',), ('3',), (None,)], ['data'])
df.toPandas()

df.show(5)

socios.limit(5).toPandas()

socios.limit(5).show()



socios.select([f.count(f.when(f.isnull(c), 1)).alias(c) for c in socios.columns]).show(10) #conta quantos valores nulos em cada coluna

socios.na.fill(0).limit(5).toPandas() #altera os valores NaN para 0

socios.na.fill('-').limit(5).toPandas() #altera os valores None para '-'

socios\
  .select('nome_do_socio_ou_razao_social', 'faixa_etaria', f.year('data_de_entrada_sociedade').alias('ano_de_entrada'))\
  .orderBy('ano_de_entrada', ascending = False)\
  .show(5, False)

socios\
  .select('nome_do_socio_ou_razao_social', 'faixa_etaria', f.year('data_de_entrada_sociedade').alias('ano_de_entrada'))\
  .orderBy(['ano_de_entrada', 'faixa_etaria'], ascending = [False, False])\
  .show(5, False)

empresas\
  .where("capital_social_da_empresa==50")\
  .show(5, False)

socios\
  .select('nome_do_socio_ou_razao_social')\
  .filter(socios.nome_do_socio_ou_razao_social.startswith('RODRIGO'))\
  .filter(socios.nome_do_socio_ou_razao_social.endswith('DIAS'))\
  .limit(10)\
  .toPandas()

df = spark.createDataFrame([('RESTAURANTE DO RUI',), ('Juca restaurantes ltda',), ('Joca Restaurante',)], ['data'])
df.toPandas()

df\
  .where(f.upper(df.data).like('%RESTAURANTE%'))\
  .show(truncate = False)

empresas\
  .select(['razao_social_nome_empresarial', 'natureza_juridica', 'porte_da_empresa', 'capital_social_da_empresa'])\
  .where(f.upper(empresas['razao_social_nome_empresarial']).like('%RESTAURANTE%'))\
  .show(truncate = False)

socios\
  .select(f.year('data_de_entrada_sociedade').alias('ano_de_entrada'))\
  .where('ano_de_entrada >= 2010')\
  .groupBy('ano_de_entrada')\
  .count()\
  .orderBy('ano_de_entrada', ascending = True)\
  .show()

empresas\
  .select('cnpj_basico', 'porte_da_empresa', 'capital_social_da_empresa')\
  .groupBy('porte_da_empresa')\
  .agg(
      f.avg('capital_social_da_empresa').alias('capital_social_medio'),
      f.count('cnpj_basico').alias('frequencia')

  )\
  .orderBy('porte_da_empresa', ascending = True)\
  .show()

empresas\
  .select('capital_social_da_empresa')\
  .summary()\
  .show()

produtos = spark.createDataFrame([
        ('1', 'Bebidas', 'Água mineral'),
        ('2', 'Limpeza', 'Sabão em pó'),
        ('3', 'Frios', 'Queijo'),
        ('4', 'Bebidas', 'Refrigerante'),
        ('5', 'Pet', 'Ração para cães')
    ],
    ['id', 'cat', 'prod'])

produtos.toPandas()

impostos = spark.createDataFrame( [
        ('Bebidas', 0.15),
        ('Limpeza', 0.05),
        ('Frios', 0.065),
        ('Carnes', 0.08)
    ],
    ['cat', 'tax'])

impostos.toPandas()

produtos.join(impostos, 'cat', how = 'inner')\
.sort('id')\
.show()

produtos.join(impostos, 'cat', how = 'left')\
.sort('id')\
.show()

produtos.join(impostos, 'cat', how = 'right')\
.sort('id')\
.show()

produtos.join(impostos, 'cat', how = 'outer')\
.sort('id')\
.show()

empresas_join = empresas.join(estabelecimentos, 'cnpj_basico', how = 'inner')
empresas_join.printSchema()

freq = empresas_join\
          .select('cnpj_basico', f.year('data_de_inicio_atividade').alias('data_de_inicio'))\
          .where('data_de_inicio >= 2010')\
          .groupBy('data_de_inicio')\
          .agg(f.count('cnpj_basico').alias('frequencia'))\
          .orderBy('data_de_inicio', ascending = True)

freq.toPandas()

freq.union(
    freq.select(
        f.lit('Total').alias('data_de_inicio'),
        f.sum(freq.frequencia).alias('frequencia')
    )
).show()

empresas.createOrReplaceTempView('empresasView')

spark.sql("SELECT * FROM empresasView")\
  .show(5)

spark\
  .sql('''
  SELECT *
  FROM empresasView
  WHERE capital_social_da_empresa = 50
  ''')\
  .show(5)

spark\
  .sql('''
  SELECT porte_da_empresa, MEAN(capital_social_da_empresa) AS MEDIA
  FROM empresasView
  GROUP BY porte_da_empresa
  ''')\
  .show(5)

empresas_join.createOrReplaceTempView('empresasJoinView')

freq = spark\
    .sql("""
        SELECT YEAR(data_de_inicio_atividade) AS data_de_inicio, COUNT(cnpj_basico) AS count
            FROM empresasJoinView
            WHERE YEAR(data_de_inicio_atividade) >= 2010
            GROUP BY data_de_inicio
            ORDER BY data_de_inicio
    """)

freq\
    .show()

freq.createOrReplaceTempView('freqView')

#juntando dataframe um embaixo do outro usando sql no spark

spark\
  .sql('''
  SELECT *
  FROM freqView
  UNION ALL
  SELECT 'Total' AS data_de_inicio, SUM(count) AS co
  FROM freqView
  ''')\
  .show(5)

  #cria uma linha artificial ('Total') na coluna data_de_inicio

empresas.write.csv(
    path = '',
    mode = 'overwrite',
    sep = ';',
    header = True
)

empresas2 = spark.read.csv(
    '',
    sep = ';',
    inferSchema = True,
    header = True
)

socios.write.csv(
    path = '',
    mode = 'overwrite',
    sep = ';',
    header = True
)

estabelecimentos.write.csv(
    path = '',
    mode = 'overwrite',
    sep = ';',
    header = True
)

empresas2.printSchema()

empresas.write.parquet(
    path = '',
    mode = 'overwrite'
)

empresas_parquet = spark.read.parquet(
    ''
)

empresas.coalesce(1).write.csv(
    path = '',
    mode = 'overwrite',
    sep = ';',
    header = True
)

empresas.write.parquet(
    path = '',
    mode = 'overwrite',
    partitionBy = 'porte_da_empresa'
)

spark.stop()
