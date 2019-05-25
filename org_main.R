## Working directory 
# setwd("YOUR_WORKSPACE_DIRECTORY")

#### 0. Preparation ####

## Installing packages
if (!require("haven")) install.packages("haven")
if (!require("randomForest")) install.packages("randomForest")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("devtools")) install.packages("devtools")
require(devtools)
if (!require("xgboost")) install_version("xgboost", version = "0.4-4", repos = "http://cran.nexr.com/")
if (!require("ade4")) install.packages("ade4")
if (!require("data.table")) install.packages("data.table")


## Activating packages
library(haven)
library(randomForest)
library(xgboost)
library(ggplot2)
library(ade4)
library(data.table)



#### 1. Reading and preprocessing the data ####

## Reading data files
## (You must check the path of directory containing data files)
table_20 = read_sas("data/hira_200table_2013.sas7bdat")
table_30 = read_sas("data/hira_300table_2013.sas7bdat")
table_40 = read_sas("data/hira_400table_2013.sas7bdat")
table_53 = read_sas("data/hira_530table_2013.sas7bdat")


## Preprocessing diagnosis codes => We use only the first 3 codes.
table_20 = cbind(table_20, 
                 MSICK_CD_head = substr(table_20$MSICK_CD, 1, 3),
                 SSICK_CD_head = substr(table_20$SSICK_CD, 1, 3))


## Target diagnosis code: I10
idx = (table_20$MSICK_CD_head=='I10' | table_20$SSICK_CD_head=='I10')
table_20_I10 = table_20[idx, ]


## Making dataset named 'I10_merged' by joining 'table_20' and 'table_30' (we use 'left outer join'.)
I10_merged = merge(x=table_20_I10, y=table_30, by="key", all.x=TRUE)
summary(I10_merged) # Check your merged data.
str(I10_merged)     # Check your merged data.



#### 2. Training the random forest model ####

## Converting chr variables to factor variables
character_vars = (lapply(I10_merged, class) == "character")
I10_merged[, character_vars] = lapply(I10_merged[, character_vars], as.factor)
str(I10_merged)     # Check your merged data. You can see that all chr variables are changed to 'factors'.


## Dropping levels of each factor variable
I10_merged = droplevels(I10_merged)
str(I10_merged)     # Check whether the number of levels of each factor variable is smaller.


## Deleting rows containing NA(s)
I10_merged = na.omit(I10_merged)
str(I10_merged)     # Check the number of points.


## Determining unused variables
unused = c('key', 'no', 'LN_NO', 'MSICK_CD', 'SSICK_CD', 'RECU_FR_DD', 
           'RECU_TO_DD', 'RECU_DDCNT', 'VST_DDCNT', 'yno', 'DIV_CD', 'GNL_NM_CD')
I10_merged = I10_merged[, !(names(I10_merged) %in% unused)]


## Extracting x and y
y = I10_merged$DGRSLT_TP_CD
summary(y)
x =  subset(I10_merged, select=-c(DGRSLT_TP_CD))


## Fitting random forest
rf = randomForest(x, y, ntree=100)
print(rf)



#### 3. Training the XGBoost model ####

## One-hot encoding for x
factor_names = names(Filter(is.factor, x))
for (f in factor_names){
  x_all_dummy = acm.disjonctif(x[f])
  x[f] = NULL
  x = cbind(x, x_all_dummy)
}


## Mapping labels
levels(y)
levels(y) = c(0, 1, 2)
levels(y)

## Fitting XGBoost
xgb_params = list(
  objective="multi:softmax",     # multi-class classification
  num_class=3,                   # number of classes
  eta=0.1,                      # learning rate
  max.depth=4,                   # max tree depth
  eval_metric="auc"              # evaluation/loss metric
)

xgb = xgboost(data=as.matrix(x), label=as.matrix(y),
              params=xgb_params,
              nrounds=500,         # max number of trees to build
              verbose=TRUE,
              print.every.n=10
              # early.stop.round=100  # stop if no improvement within 100 trees
)
