# Algoritmos Genéticos em R

# ***** Esta é a versão 2.0 deste script, atualizado em 02/07/2017 *****
# ***** Esse script pode ser executado nas versões 3.3.1, 3.3.2, 3.3.3 e 3.4.0 da linguagem R *****
# ***** Recomendamos a utilização da versão 3.4.0 da linguagem R *****

# genalg: R Based Genetic Algorithm
# https://cran.r-project.org/web/packages/genalg/index.html
install.packages("genalg")
library(genalg)
library(ggplot2)

# Dataset e contraint de peso
dataset <- data.frame(item = c("canivete", "arroz", "batatas", "cebolas", "mochila", "corda", "compasso"), 
                      survivalpoints = c(10, 20, 15, 2, 30, 10, 30), 
                      weight = c(1, 5, 10, 1, 7, 5, 1))
weightlimit <- 20

# Antes de criar o modelo, temos de configurar uma função de avaliação. 
# A função de avaliação avaliará os diferentes indivíduos (cromossomos) da população sobre o valor da sua 
# configuração genética.

# Um indivíduo pode, por exemplo, ter a seguinte configuração de gene: 1001100.

# Cada número nesta cadeia binária representa se deve ou não levar um item consigo. 
# Um valor de 1 refere-se a colocar o item específico na mochila enquanto um 0 se refere a deixar o item em casa. 
# Dado o exemplo de configuração do gene, tomamos os seguintes itens;
chromosome = c(1, 0, 0, 1, 1, 0, 0)

# Acima, atribuímos um valor à configuração do gene de um determinado cromossomo. 
# Isso é exatamente o que a função de avaliação faz.

# O algoritmo genalg tenta otimizar para o valor mínimo. 
# Portanto, o valor é calculado como acima e multiplicado com -1. 
# Uma configuração que excede a restrição do peso retorna um valor de 0 (um valor mais alto também pode ser atribuído).
dataset[chromosome == 1, ]

# Podemos verificar para qual quantidade de pontos de sobrevivência esta configuração atende.
cat(chromosome %*% dataset$survivalpoints)

# Função de Avaliação
evalFunc <- function(x) {
  current_solution_survivalpoints <- x %*% dataset$survivalpoints
  current_solution_weight <- x %*% dataset$weight
  
  if (current_solution_weight > weightlimit) 
    return(0) else return(-current_solution_survivalpoints)
}

# Em seguida, escolhemos o número de iterações, projetamos e executamos o modelo.
iter = 100
GAmodel <- rbga.bin(size = 7, popSize = 200, iters = iter, mutationChance = 0.01, elitism = T, evalFunc = evalFunc)
cat(summary(GAmodel, echo=TRUE))

# A melhor solução encontrada foi 1 1 0 1 1 1 1  
# Isto nos leva a levar os seguintes itens conosco em nossa viagem para a natureza.
solution = c(1, 1, 0, 1, 1, 1, 1)
dataset[solution == 1, ]

# Isso, por sua vez, nos dá o número total de pontos de sobrevivência.
cat(paste(solution %*% dataset$survivalpoints, "/", sum(dataset$survivalpoints)))

# Plot
plot(GAmodel)


