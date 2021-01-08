# Projetos com feedback
# Detecção de Fraudes no Tráfego de Cliques em Propagandas de Aplicações Mobile 

# Pacotes necessários
library(kableExtra)
library(lubridate)
library(Amelia)
library(dplyr)
library(ggplot2)
library(viridis)
library(gridExtra)
library(paletteer)
library(plotrix)
library(caTools)
library(DMwR)
library(randomForest)
library(forcats)
library(e1071)
library(caret)
library(ROCR)

# Carregando datasets
# salvando o dataset original caso seja necessário futuramente
df_original <- read.csv('train_sample.csv', sep = ",", header = TRUE, stringsAsFactors = FALSE)

# carregando o dataset para trabalho
df <- read.csv('train_sample.csv', sep = ",", header = TRUE, stringsAsFactors = FALSE)

# Pré-processamento de dados
# estrutura do dataset antes do pré-processamento
str(df)
# ip = retirado (identificador único)
df$ip <- NULL
# alterando tipos das colunas
# app, device, os, channel e is_attributed = factor
fatores <- c('app', 'device', 'os', 'channel', 'is_attributed')
for (i in colnames(df)) {
    if (i %in% fatores) {
        df[,i] <- as.factor(df[,i])
    }
}

# click_time e attributed_time = data
df$click_time <- ymd_hms(df$click_time)
df$attributed_time <- ymd_hms(df$attributed_time)

# criação de nova coluna para análise futura
# duration = hora do download - hora do clique
df$duration <- as.numeric(df$attributed_time - df$click_time)
df$duration[is.na(df$duration)] <- 0

# verificando dados missing
missmap(df, 
        main = "TalkingData - Detecção de cliques fraudulentos - Mapa de Dados Missing",
        col = c("yellow", "black"), 
        legend = FALSE)
# valores missing na coluna attributed_time existem pois estes valores estão atrelados à coluna is_attributed. Se esta última for 0, a primeira não é preenchida.
 
# estrutura do dataset após pré-processamento
str(df)

# Análise exploratória

# tops de cliques (variáveis categóricas)
# (gráficos de colunas em grid)

plot_app <- df %>%
    group_by(app) %>%
    count(sort = T) %>%
    head(5) %>%
    ggplot(aes(x = reorder(app, -n), y = n, fill = -n)) +
    geom_col(show.legend = F) +
    labs(title = 'Mais cliques - Aplicativos',
         x = 'App',
         y = NULL) +
    scale_fill_viridis(option = "viridis",direction = 1) +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5))

plot_device <- df %>%
    group_by(device) %>%
    count(sort = T) %>%
    head(5) %>%
    ggplot(aes(x = reorder(device, -n), y = n, fill = -n)) +
    geom_col(show.legend = F) +
    labs(title = 'Mais cliques - Dispositivos',
         x = 'Devices',
         y = NULL) +
    scale_fill_viridis(option = "viridis",direction = 1) +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5))

plot_os <- df %>%
    group_by(os) %>%
    count(sort = T) %>%
    head(5) %>%
    ggplot(aes(x = reorder(os, -n), y = n, fill = -n)) +
    geom_col(show.legend = F) +
    labs(title = 'Mais cliques - Sistemas Operacionais',
         x = 'OS',
         y = NULL) +
    scale_fill_viridis(option = "viridis",direction = 1) +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5))

plot_channel <- df %>%
    group_by(channel) %>%
    count(sort = T) %>%
    head(5) %>%
    ggplot(aes(x = reorder(channel, -n), y = n,fill = -n)) +
    geom_col(show.legend = F) +
    labs(title = 'Mais cliques - Canais',
         x = 'Channels',
         y = NULL) +
    scale_fill_viridis(option = "viridis",direction = 1) +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5))

grid.arrange(plot_app, plot_device, plot_os, plot_channel,
             ncol = 2,
             nrow = 2)

# série temporal, plotada em facet por dia, do horário que houve mais cliques
# (gráfico de pontos + linhas em facet_grid)
df %>%
    mutate(dia_da_semana = wday(click_time, label = T)) %>%
    mutate(hora =hour(click_time)) %>%
    group_by(hora, dia_da_semana) %>%
    count() %>%
    ungroup() %>%
    ggplot(aes(x = hora, y = n, color = dia_da_semana)) + 
    geom_point(show.legend = F) + geom_line(show.legend = F) +
    facet_grid(~ dia_da_semana) +
    labs(title = 'Cliques por dia e horário',
         y = NULL) +
    theme(plot.title = element_text(hjust = 0.5))

# Média de tempo de conversão clique-download em segundos para cada dia da semana
# (gráfico de barras)

df %>%
    filter(is_attributed == 1) %>%
    mutate(dia_da_semana = wday(attributed_time, label = T)) %>%
    group_by(dia_da_semana) %>%
    summarise(media = mean(duration)) %>%
    ungroup() %>%
    ggplot(aes(x = reorder(dia_da_semana, media), y = media, fill = dia_da_semana)) +
    geom_col(show.legend = F) + coord_flip() +
    labs(title = 'Média de tempo de conversão em download',
         subtitle = 'em segundos para cada dia da semana',
         x = NULL, y = NULL) +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5))

# cliques que se converteram em downloads (1) x que não se converteram (0)
# (gráfico de pizza)

cores_2 <- as.vector(paletteer_c("viridis::viridis", n = 2, direction = -1))

tabela <- table(df$is_attributed)
tabela <- prop.table(tabela) * 100

download <- c('Sim', 'Não')
labels <- paste(download, tabela)
labels <- paste(labels, '%', sep = '')

pie3D(tabela, labels = labels,
      col = cores_2,
      main = 'Cliques que se converteram em downloads',
      theta = pi/4)

# Machine Learning

# SPLIT TREINO E TESTE
# Utilizando a função sample.split do pacote caTools dividimos o dataset em treino e teste (proporção de 70-30 %)

df_split <- df[, -c(5, 6, 8)]

indice <- sample.split(df_split$is_attributed, SplitRatio = .7)

treino <- subset(df_split, indice == T)
teste <- subset(df_split, indice == F)

# obs: as colunas de datas precisaram ser retiradas para que o balanceamento pudesse ser feito posteriormente.

# BALANCEAMENTO DO DATASET
# Identificamos a existência muito maior de cliques fraudulentos, o que pode prejudicar o desempenho do algoritmo (99,8% para 0,2%, aproximadamente).
# Dessa forma, vamos aplicar uma técnica de balanceamento chamada SMOTE, para que a desigualdade na proporção seja menor:
# rapidamente vamos verificar o desbalanceamento após separação em dados de treino e teste

d_treino <- as.vector(round(prop.table(table(treino$is_attributed)) * 100 ,4))
d_teste <- as.vector(round(prop.table(table(teste$is_attributed)) * 100, 4))

desbalanceados <- data.frame(treino = paste(d_treino, '%'),
                             teste = paste(d_teste, '%'),
                             row.names = c('is_attributed = 0 (apps não baixados)',
                                           'is_attributed = 1 (apps baixados)'))

treino_smote <- SMOTE(is_attributed ~ ., data = treino)

b_treino <- as.vector(round(prop.table(table(treino_smote$is_attributed)) * 100, 4))

balanceados <- data.frame(proporcao = paste(b_treino, '%'),
                          row.names = c('is_attributed = 0 (apps não baixados)',
                                        'is_attributed = 1 (apps baixados)'))

# FEATURE SELECTION
# Utilizando o algoritmo random forest, vamos identificar as variáveis mais relevantes para o treinamento do modelo
# funções: random forest (algoritmo) + varImPlot (gráfico)

# antes de rodar o modelo, vamos reduzir os levels das variáveis fatores tendo em vista que o radom forest não roda com variáveis categóricas que contenham mais de 53 níveis.
# obs: foi necessário reduzir mais ainda as categorias (de 53 para 40) para que a função pudesse fazer os agrupamentos

aux1 <- treino_smote %>%
    select(-is_attributed) %>%
    mutate(app = fct_lump(app, n = 40),
           device = fct_lump(app, n = 40),
           os = fct_lump(app, n = 40),
           channel = fct_lump(app, n = 40))

aux2 <- treino_smote %>%
    select(is_attributed)

treino_fs <- cbind(aux1, aux2)

set.seed(1234)
modelo_fs <- randomForest(is_attributed ~ .,
                          data = treino_fs,
                          importance = T)
varImpPlot(modelo_fs,
           main = 'Feature Selection com Random Forest')

# TREINAMENTO DO MODELO
set.seed(1234)
modelo <- naiveBayes(is_attributed ~ app
                     + channel
                     + os,
                     data = treino_smote)

previsao <- predict(modelo, teste)

# AVALIAÇÃO DO MODELO
# confusion matrix
# Com a função de matriz de confusão do pacote caret, analisei se o modelo teve ou não boa performance

confusionMatrix(teste$is_attributed, previsao)

# na matriz de confusão podemos observar número alto de verdadeiros positivos e negativos, bem como ótima acurácia
 
# curva ROC
# para ilustrar, também podemos gerar a curva ROC
# Receiver Operating Characteristic
# taxa verdadeiro positivo = sensibilidade = quantas vezes o modelo acertou para a opção positiva do classificador (0 - não fez o download, clique fraudulento)
# conta: 28844 / (28844 + 1088) = 96,37%
# taxa falso positivo = especificidade (1) = quantas vezes o modelo errou, classificando como clique fraudulento, porém, era um download realizado (1)
# conta: 11 / (11 + 53) = 17,18%
# quanto maior a taxa de verdadeiro positivo e menor a taxa de falso positivo, melhor para o modelo. Ou seja: o canto superior esquerdo é ponto ótimo
# objetivo = detectar o máximo possível de verdadeiros positivos, enquanto minimiza os falsos positivos

pred <- prediction(as.numeric(previsao), teste$is_attributed)
perf <- performance(pred, "tpr","fpr") 
plot(perf, col = 'red',
     main = 'Curva ROC',
     ylab = 'Sensibilidade (96,37%)',
     xlab = '1-Especificidade (17,18%)')

# NOVO TREINAMENTO?
# acurácia alta e ROC curve mostraram bons resultados - não testarei outro algoritmo