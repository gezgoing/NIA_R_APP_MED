## Working directory 
# setwd("YOUR_WORKSPACE_DIRECTORY")

#### 0. 패키지 준비 ####

## 패키지 설치 (p.27~28)
if (!require("haven")) install.packages("haven")
if (!require("randomForest")) install.packages("randomForest")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("devtools")) install.packages("devtools")  # (p.28)
require(devtools)  # (p.28)
if (!require("xgboost")) install_version("xgboost", version = "0.4-4", repos = "http://cran.nexr.com/")  # (p.28)
if (!require("ade4")) install.packages("ade4")
if (!require("data.table")) install.packages("data.table")

## 패키지를 메모리에 로드 (p.29)
library(haven) # sas7bdat 파일 읽기
library(randomForest) # 랜덤 포레스트 모델링 패키지
library(xgboost) # XGboost 모델링 패키지
library(ggplot2) # 시각화를 위한 패키지
library(ade4) # one-hot encoding을 위한 패키지
library(data.table) # 데이터 테이블 패키지



#### 1. 데이터 읽기와 정제 ####

## SAS 데이터 읽기 (p.29)
## 아래의 데이터 파일 이름 앞의 path 확인 (data/)
table_20 = read_sas("data/hira_200table_2013.sas7bdat") # 명세서 일반내역
table_30 = read_sas("data/hira_300table_2013.sas7bdat") # 진료내역
table_40 = read_sas("data/hira_400table_2013.sas7bdat") # 상병내역
table_53 = read_sas("data/hira_530table_2013.sas7bdat") # 처방전 상세내역

## 질병분류코드 코드의 앞 3자리만 사용 (p.30)
# http://www.koicd.kr/2016/kcd/v7.do#0&n
head(data.frame(table_20),2)
table_20 = cbind(table_20, 
                 MSICK_CD_head = substr(table_20$MSICK_CD, 1, 3), # 주상병코드
                 SSICK_CD_head = substr(table_20$SSICK_CD, 1, 3)) # 부상병코드
# Alternatively,
table_20$MSICK_CD_head = substr(table_20$MSICK_CD, 1, 3)
table_20$SSICK_CD_head = substr(table_20$SSICK_CD, 1, 3)

## 타겟코드: I10 (본태성(원발성) 고혈압을 의미) (p.31~33)
# http://www.koicd.kr/2016/kcd/v7.do#9.3.1&n
idx = (table_20$MSICK_CD_head=='I10' | table_20$SSICK_CD_head=='I10')
table_20_I10 = table_20[idx, ]
head(table_20_I10,2)

## table_30: 진료내역(투약량, 총 투약량, 단가, 금액 등)
## table_20과 table_30을 결합
I10_merged = merge(x=table_20_I10, y=table_30, by="key", all.x=TRUE)
summary(I10_merged) # 결합 데이터 확인
str(I10_merged)     # 결합 데이터 확인



#### 2. 랜덤 포레스트 모델 학습 #### (p.34)

## 문자형 변수를 범주형 변수로 변환 (알고리즘이 문자형 변수를 다루지 못하기 때문)
character_vars = (lapply(I10_merged, class) == "character")
I10_merged[, character_vars] = lapply(I10_merged[, character_vars], as.factor)
str(I10_merged)     # 문자형 변수 모두가 범주형으로 바뀌었는지 확인

## 범주형 변수의 Level 정리 (p.35)
I10_merged = droplevels(I10_merged)
str(I10_merged)     # 정리 여부 확인

## NA가 존재하는 행 제거
I10_merged = na.omit(I10_merged)
str(I10_merged)     # 변수별 항목 수 확인

## [설명] 미사용 변수 결정 (p.36에서 설명)
unused = c('key', 'no', 'LN_NO', 'MSICK_CD', 'SSICK_CD', 'RECU_FR_DD', 
           'RECU_TO_DD', 'RECU_DDCNT', 'VST_DDCNT', 'yno', 'DIV_CD', 'GNL_NM_CD')
I10_merged = I10_merged[, !(names(I10_merged) %in% unused)]
str(I10_merged)

## 훈련/테스트 데이터 분리
set.seed(111)
ind<-sample(2,nrow(I10_merged),prob=c(0.75,0.25),replace=T)

tr <- I10_merged[ind==1,]
tt <- I10_merged[ind==2,]

## 타겟변수와 설명변수 생성 (p.37)
y = tr$DGRSLT_TP_CD
summary(y)
x =  subset(tr, select=-c(DGRSLT_TP_CD))

summary(tt$DGRSLT_TP_CD)

# 1:계속, 5:기타, 9:퇴원 (2:이송, 3:회송, 4:사망)


## 파라미터 설명
## ntree: 학습할 트리 모델 수
## mtry: 하나의 트리 모델이 사용할 랜덤 변수의 수
##       (clf: sqrt(#var), reg: #var/3)

## 랜덤포레스트 알고리즘 학습 (p.38)
rf = randomForest(x, y, ntree=100)
print(rf)

table(predict(rf,tt,type='response'),tt$DGRSLT_TP_CD)

# 중요 변수 시각화
varImpPlot(rf)
str(rf)
str(rf,max.level=1)
rf$importance



#### 3. XGBoost 모델 학습 #### (p.39)
# https://brunch.co.kr/@snobberys/137

## 범주형 변수 --> One-hot encoding (알고리즘이 숫자형만 처리 가능)
factor_names = names(Filter(is.factor, x))
for (f in factor_names){
  x_all_dummy = acm.disjonctif(x[f])
  x[f] = NULL
  x = cbind(x, x_all_dummy)
}


## 타겟변수도 0부터 시작하는 범주형으로 변환(1-->0, 5-->1, 9-->2)
levels(y)
summary(y)

levels(y) = c(0, 1, 2)
levels(y)
summary(y)

## XGBoost 모델 학습 (p.41)
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


# ---------------------------------

## 범주형 변수 --> One-hot encoding (알고리즘이 숫자형만 처리 가능)
x_ <- I10_merged[,c(1:9,11:37)]
factor_names = names(Filter(is.factor, x_))
for (f in factor_names){
  x_all_dummy = acm.disjonctif(x_[f])
  x_[f] = NULL
  x_ = cbind(x_, x_all_dummy)
}


## 타겟변수도 0부터 시작하는 범주형으로 변환(1-->0, 5-->1, 9-->2)
y_<-I10_merged$DGRSLT_TP_CD
levels(y_)
levels(y_) = c(0, 1, 2)
levels(y_)

dim(x_)
length(y_)

## 훈련/테스트 데이터 분리
table(ind)

x_tr<-x_[ind==1,]
y_tr<-y_[ind==1]
sum(ind==1);nrow(x_tr);length(y_tr)

x_tt<-x_[ind==2,]
y_tt<-y_[ind==2]
sum(ind==2);nrow(x_tt);length(y_tt)


## XGBoost 모델 학습 (p.41)
xgb_params1 = list(
  objective="multi:softmax",     # multi-class classification
  num_class=3,                   # number of classes
  eta=0.3,                      # learning rate
  max.depth=5,                   # max tree depth
  eval_metric="auc"              # evaluation/loss metric
)

xgb1 = xgboost(data=as.matrix(x_tr), label=as.matrix(y_tr),
              params=xgb_params1,
              nrounds=500,         # max number of trees to build
              verbose=TRUE,
              print.every.n=10
              # early.stop.round=100  # stop if no improvement within 100 trees
)

pred_tr <- predict (xgb1,as.matrix(x_tr))
table(pred_tr,y_tr)

pred_tt <- predict (xgb1,as.matrix(x_tt))
table(pred_tt,y_tt)
