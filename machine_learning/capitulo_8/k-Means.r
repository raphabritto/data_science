# Clustering com K-Means
# https://cran.r-project.org/web/views/Cluster.html

# ***** Esta é a versão 2.0 deste script, atualizado em 02/07/2017 *****
# ***** Esse script pode ser executado nas versões 3.3.1, 3.3.2, 3.3.3 e 3.4.0 da linguagem R *****
# ***** Recomendamos a utilização da versão 3.4.0 da linguagem R *****

# Identificando segmentos de adolescentes
# Dataset com dados de 30.000 adolescentes dos EUA 

# Definindo o diretório de trabalho
setwd("~/Dropbox/DSA/MachineLearning/R/Cap08")
getwd()
sessionInfo()
RStudio.Version()

# Carregando os dados
dados <- read.csv("dados_adolescentes.csv")
str(dados)
head(dados)

# Buscando valores missing para variáveis relacionadas ao sexo 
table(dados$gender)
table(dados$gender, useNA = "ifany")

# Buscando valores missing para variáveis relacionadas a idade 
summary(dados$age)

# Eliminando outliers na variável idade
dados$age <- ifelse(dados$age >= 13 & dados$age < 20, dados$age, NA)

summary(dados)

# Atribuindo valores missing na variável sexo para o tipo "outros"
dados$female <- ifelse(dados$gender == "F" & !is.na(dados$gender), 1, 0)
dados$outros <- ifelse(is.na(dados$gender), 1, 0)

# Verificando o resultado
table(dados$gender, useNA = "ifany")
table(dados$female, useNA = "ifany")
table(dados$outros, useNA = "ifany")

# Buscando a média de idade
mean(dados$age, na.rm = TRUE)

# Agregando os dados e calculando a média de idade por ano em que estev na escola
?aggregate
aggregate(data = dados, age ~ gradyear, mean, na.rm = TRUE)

# Cria um vetor com a média de idade para cada ano escolar
media_idade <- ave(dados$age, dados$gradyear, FUN = function(x) mean(x, na.rm = TRUE))
media_idade
dados$age <- ifelse(is.na(dados$age), media_idade, dados$age)


## Preparando o modelo

# Criando um vetor com os interesses de cada jovem
interesses <- dados[5:40]

# Normalizando os dados
interesses_z <- as.data.frame(lapply(interesses, scale))

# Criando o modelo
set.seed(2345)
?kmeans
dados_clusters <- kmeans(interesses_z, 5)
print(dados_clusters)

# Verificando o tamanho dos clusters
dados_clusters$size

# Verificando o centro dos clusters
dados_clusters$centers

# Aplicando o ID dos clusters ao dataframe original
dados_clusters$cluster
dados$cluster <- dados_clusters$cluster

# Verificando os 5 primeiros registros
dados[1:5, c("cluster", "gender", "age", "friends")]

# Média de idade por cluster
aggregate(data = dados, age ~ cluster, mean)

# Proporção de mulheres por cluster
aggregate(data = dados, female ~ cluster, mean)

# Média de número de amigos por cluster
aggregate(data = dados, friends ~ cluster, mean)


## Visualização e Comparação de Modelos

# Pacotes
install.packages("flemix")
install.packages("fpc")
library(cluster)
library(fpc)

# Visualizando os clusters
?plotcluster
plotcluster(interesses, dados$cluster)

# Comparando 2 modelos de clustering

# Cria outro modelo K-Means
dados_clusters2 <- kmeans(interesses_z, 6)

# Crias a matriz de distâncias
d = dist(dados)

# Compara os 2 modelos
??cluster.stats
dist(dados)
cluster.stats(d, dados$cluster, dados_clusters2$cluster)
