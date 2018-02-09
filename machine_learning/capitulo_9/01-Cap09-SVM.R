# Classificação Multiclasse com SVM

# ***** Esta é a versão 2.0 deste script, atualizado em 02/07/2017 *****
# ***** Esse script pode ser executado nas versões 3.3.1, 3.3.2, 3.3.3 e 3.4.0 da linguagem R *****
# ***** Recomendamos a utilização da versão 3.4.0 da linguagem R *****

# Pacote
install.packages("e1071")
library(e1071)

# Help
help(svm)

# Dataset
data(iris)
attach(iris)
head(iris)

# Criação do modelo
modelo <- svm(Species ~ ., data = iris)
print(modelo)

# Alternativamentem, podemos usar esta sintaxe
x <- subset(iris, select = -Species)
y <- Species
modelo <- svm(x, y) 
print(modelo)
summary(modelo)

# Previsões
pred <- predict(modelo, x)
pred <- fitted(modelo)

# Checando a acurácia
table(pred, y)

# Computando valores de decisão e probabilidades
pred <- predict(modelo, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]

# Plot (classes em círculos coloridos e Vetores de Suporte como cruzes:
plot(cmdscale(dist(iris[,-5])),
     col = as.integer(iris[,5]),
     pch = c("o","+")[1:150 %in% modelo$index + 1])


# Ajuste dos parâmetros
# Valores de gamma testados" 0.5,1,2
# Valores de cost testados 1/10 , 10 , 100
?tune
tune.results <- tune(svm, 
                     train.x = iris[1:4], 
                     train.y = iris[,5], 
                     kernel = 'radial',
                     ranges = list(cost = 10^(-1:2), gamma = c(.5,1,2)))
  
summary(tune.results)

tuned.svm <- svm(Species ~ ., data = iris, kernel = "radial", cost = 1, gamma = 0.5)

summary(tuned.svm)
  
tuned.predicted.values <- predict(tuned.svm,iris[1:4])
table(tuned.predicted.values,iris[,5])


# Modelo SVM para Análise de Crédito

library(e1071) 
library(caret) 
# Otimização de Hiperparametros
library(kernlab) 
library(ROCR) 

# Dataset
credit.df <- read.csv("clientes.csv", header = TRUE, sep = ",")
head(credit.df)

# Função para Fatorização
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

# Função para Normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalizando
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
credit.df <- scale.features(credit.df, numeric.vars)

# Fatorizando
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')

credit.df <- to.factors(df = credit.df, variables = categorical.vars)
str(credit.df)

# Dados de treino e de teste, a taxa de 60:40
indexes <- sample(1:nrow(credit.df), size=0.6*nrow(credit.df))
train.data <- credit.df[indexes,]
test.data <- credit.df[-indexes,]

# Separando atributos e variável target
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

# Construindo o modelo com dados de treino
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
svm.model <- svm(formula = formula.init, data = train.data, kernel = "radial", cost = 100, gamma = 1)

# Visualizando o resultado
summary(svm.model)

# Prevendo e avaliando resultados
svm.predictions <- predict(svm.model, test.feature.vars)
confusionMatrix(data = svm.predictions, reference = test.class.var, positive = "1")


# Feature Selection
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula.init, data = train.data, method = "svmRadial", trControl = control)
importance <- varImp(model, scale = FALSE)
plot(importance, cex.lab = 0.5)


# Construindo o modelo com as variáveis selecionadas
formula.new <- "credit.rating ~ account.balance + credit.duration.months + previous.credit.payment.status + credit.amount"
formula.new <- as.formula(formula.new)
svm.model.new <- svm(formula = formula.new, data = train.data, kernel = "radial", cost = 10, gamma = 0.25)

# Previsões
svm.predictions.new <- predict(svm.model.new, test.feature.vars)

# Avaliação do mnovo modelo
confusionMatrix(data = svm.predictions.new, reference = test.class.var, positive = "1")

# Otimização dos Hiperparâmetros

# Grid Search
cost.weights <- c(0.1, 10, 100)
gamma.weights <- c(0.01, 0.25, 0.5, 1)
tuning.results <- tune(svm, 
                       formula.new, 
                       data = train.data, 
                       kernel = "radial", 
                       ranges = list(cost = cost.weights, gamma = gamma.weights))

# Resultado da otimização
print(tuning.results)

# Plot
plot(tuning.results, cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")

# Verificando a acurácia do modelo após a otimização dos hiperparâmetros
svm.model.best = tuning.results$best.model
svm.predictions.best <- predict(svm.model.best, test.feature.vars)
confusionMatrix(data = svm.predictions.best, reference = test.class.var, positive = "1")


# Plot 
plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf,col="black",lty=1, lwd=2,
       main=title.text, cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")
  abline(0,1, col="red")
  auc <- performance(predictions,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend=c(paste0("AUC: ",auc)),cex=0.6,bty = "n",box.col = "white")
  
}
svm.predictions.best <- predict(svm.model.best, test.feature.vars, decision.values = T)
svm.prediction.values <- attributes(svm.predictions.best)$decision.values
predictions <- prediction(svm.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="Curva ROC SVM")

