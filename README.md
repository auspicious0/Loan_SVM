# 대출금 상환 여부 (SVM)

본 문서는 Kaggle의 데이터셋을 활용하여 데이터 분석 및 SVM 을 통한 분류 모델 구축을 목표로 합니다. 이 데이터셋은 대출금 상환 여부에 관련된 정보를 담고 있습니다. 데이터를 적절하게 전처리한 후, 모델링과 예측을 수행합니다.

## 목차
1. [패키지 설치 및 그래프 설정](#1-패키지-설치-및-그래프-설정)
2. [데이터 수집](#2-데이터-수집)
3. [데이터 전처리](#3-데이터-전처리)
   1. [결측값 처리](#3-1-결측값-처리)
   2. [데이터 변수의 형 변환](#3-2-데이터-변수의-형-변환)
   3. [중간값 처리 및 이상값 처리](#3-3-중간값-처리-및-이상값-처리)
4. [bagging 및 RandomForest 분석](#4-bagging-및-randomforest-분석)
   1. [회귀 분석](#4-1-회귀-분석)
   2. [분류 분석](#4-2-분류-분석)
5. [문의](#5-문의)





## 1. 패키지 설치 및 그래프 설정

프로젝트를 시작하기 전, 필요한 R 패키지를 설치하고 그래프 설정을 합니다.
 ```
#패키지 부착 및 출력 그래프의 크기 설정
install.packages(c("tidyverse", "caret", "e1071", "Hmisc"))
library(tidyverse)
library(data.table)

library(repr)
options(repr.plot.width = 7, repr.plot.height = 7)
```

## 2. 데이터 수집

대출금 관 데이터는 Kaggle에서 제공되며 다음 링크에서 얻을 수 있습니다:

[Kaggle Dataset](https://www.kaggle.com/datasets/sarahvch/predicting-who-pays-back-loans?select=loan_data.csv)


데이터를 다운로드하고 분석에 활용합니다.

```
#https://www.kaggle.com/datasets/sarahvch/predicting-who-pays-back-loans?select=loan_data.csv
#https://drive.google.com/file/d/1SfB1iLRv7wyKsZLMxT-2rGwPnROv0BDl/view?usp=sharing
system("gdown --id 1SfB1iLRv7wyKsZLMxT-2rGwPnROv0BDl")
system("ls",TRUE)

lo <- fread("loan_data.csv",encoding = "UTF-8") %>% as_tibble()
```

## 3. 데이터 전처리

불러온 데이터를 면밀히 확인 및 분석한  결측값 처리, 이상값 처리 등을 수행합니다. 데이터의 구조를 확인하고 필요한 변수를 팩터(factor)로 변환합니다. 

```
lo %>% show()
str(lo)
summary(lo)
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/0e5225b6-d477-4a05-955f-7b37200389be)

결측값은 없습니다.

우선 정수형 변수들 중 min 값이 0, max값이 5이하인 변수들을 unique()를 통해 factor형으로 변환이 가능한 변수인지 알아보겠습니다.

문자형 변수(purpose)와 not.fully.faid(반응, 종속변수) 역시 factor형으로 변환 가능한 변수인지 살피겠습니다.

### 3-1. 데이터 형 변환

데이터 형 변환을 위해 우선 형변환할 데이터가 변환 가능한 데이터인지 확인해 보겠습니다.

```
lo$credit.policy %>% unique()
lo$purpose %>% unique()
lo$pub.rec %>% unique()
lo$not.fully.paid %>% unique()
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/e4bd2ebb-9bd2-4277-9a8b-a00e32f2c8c3)

모두 변환 가능한 데이터 임을 확인할 수 있습니다. 

```
lo <- lo %>%
  mutate_at(c("credit.policy", "purpose", "pub.rec", "not.fully.paid"),factor)
lo %>% str()
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/5cee9824-bfda-4512-90d7-334d222c1f82)

알맞게 변환되었습니다.


### 3-2. 이상값 처리

결측값은 앞서 살펴본 대로 없기 때문에 이상값 처리를 진행하겠습니다.


```
# 이상치 및 결측값 처리 함수

calculate_outliers <- function(data, column_name) {
  iqr_value <- IQR(data[[column_name]])
  upper_limit <- summary(data[[column_name]])[5] + 1.5 * iqr_value
  lower_limit <- summary(data[[column_name]])[2] - 1.5 * iqr_value

  data[[column_name]] <- ifelse(data[[column_name]] < lower_limit | data[[column_name]] > upper_limit, NA, data[[column_name]])

  return(data)
}
table(is.na(lo))
# boxplot 을 그리기 위해 factor형 변수를 삭제해 lo_에 저장하겠습니다.
lo_ <- lo %>% select(-credit.policy, -purpose, -pub.rec, -not.fully.paid)
boxplot(lo_)
# 이상치 및 결측값 처리 및 결과에 대한 상자그림 그리기
lo <- calculate_outliers(lo, "int.rate")
lo <- calculate_outliers(lo, "installment")
lo <- calculate_outliers(lo, "log.annual.inc")
lo <- calculate_outliers(lo, "dti")
lo <- calculate_outliers(lo, "fico")
lo <- calculate_outliers(lo, "days.with.cr.line")
lo <- calculate_outliers(lo, "revol.bal")
lo <- calculate_outliers(lo, "revol.util")
lo <- calculate_outliers(lo, "inq.last.6mths")
lo <- calculate_outliers(lo, "delinq.2yrs")

table(is.na(lo))
lo <- na.omit(lo)
table(is.na(lo))
lo_ <- lo %>% select(-credit.policy, -purpose, -pub.rec, -not.fully.paid)
boxplot(lo_)
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/97c92c08-7174-4d12-88dd-cff44b5129ee)

이상값이 삭제되었습니다. 그림을 통해 살펴보겠습니다. 

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/875f63ba-9aab-4257-8405-60d88a5392bb)

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/01e4d687-aef4-46ef-903f-45ed0af40555)

이상값이 없어진 것을 직관적으로 확인할 수 있습니다.



### 3-3. 데이터 분할

데이터를 test 데이터와 train 데이터로 분할하여 train 데이터론 모델링을 진행하고 test 데이터를 통해 예측을 진행해 보겠습니다.

무작위로 데이터를 분리하지 않고 반응변수를 중심으로 8:2로 나누기 위해 caret::createDataPartition을 사용하겠습니다.

```
library(caret)
index <- caret::createDataPartition(y = lo$not.fully.paid, p = 0.8, list = FALSE)
train <- lo[index,]
test <- lo[-index,]

train %>% show()
test%>% show()
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/cb7becfa-09b8-45ef-832d-9907236b7e8c)

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/57f363fa-e73f-4bf1-867c-ca2c01838e8c)

데이터가 잘 분할 되었습니다.

## 4. SVM 분석

SVM 모델을 생성하고 분류 결과를 확인해 보겠습니다.

### 4-1. SVM  모델 생성

train데이터를 이용하여 SVM 모델을 생성하고 분류 결과를 확인해 보겠습니다.

confusionMatrix를 생성하여 실제 결과와 분류 결과를 비교하여 보겠습니다.

```
svm_basic <- e1071::svm(formula = not.fully.paid ~ ., data = train, type = "C-classification", kernel = "radial")
summary(svm_basic)

print("svm_basic : train 데이터 분류 결과")
table(predict(svm_basic, train), train$not.fully.paid)

print("svm_basic : train 데이터 confusionMatrix 결과")
cm <- caret::confusionMatrix(predict(svm_basic, train), train$not.fully.paid)
cm
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/73efb4c3-3948-4b46-9670-149e0b1061ea)

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/df2ba943-1f4d-4199-8e25-bd4f455d2e5a)


매우 정확히 분류된 것을 알 수 있습니다.

이제 해당 모델을 통해 예측을 수행해 확인해 보고 다시 confusionMatrix를 통해 해당 모델 예측의 성능지표를 확인해 보겠습니다.
```
predict_value_svm <- predict(svm_basic, test, type = "C-classification") %>%
  tibble(predict_value_svm = .)
predict_check_svm <- test %>% select(not.fully.paid) %>% dplyr::bind_cols(.,predict_value_svm)
head(predict_check_svm)
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/729216d7-d357-4403-af12-82795802aab2)

```
cm <- caret::confusionMatrix(predict(svm_basic, test), test$not.fully.paid)
cm
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/148c007d-ed28-4a6d-90ed-aa505080dd7a)


86퍼센트이 정확도를 확인해 볼 수 있습니다.

이제 하이퍼 파라메터를 조정해보겠습니다.

### 4-2. 하이퍼 파라메터 

곡률(gamma)과 마진 폭(cost)을 결정하기 위해 튜닝 작업을 수행하겠습니다.

gamma는 10^(-8:1), cost는 1~30 범위로 총 300개 조합으로 튜닝을 진행해 최적의 하이퍼 파라미터를 찾아 저장하려했으나 데이터의 양이 많아 튜닝이 진행되지 않습니다. 따라서 더 작은 범위로 축소하겠습니다. (10^(-3:1), cost = 1:10, 30개 조합)


또 병렬 처리를 진행하겠습니다. tunecontrol 매개변수를 사용하여 병렬 처리를 활성화 해보겠습니다.

```
install.packages("doParallel")
library(doParallel)
registerDoParallel(cores = 4)
tuned <- e1071::tune.svm(not.fully.paid ~ ., data = train, gamma = 10^(-8:1),cost = 1:30)
tuned <- e1071::tune.svm(not.fully.paid ~ ., data = train, gamma = 10^(-3:1),cost = 1:10)
```

하지만 2시간 넘게 진행되지 않았습니다. 데이터 양이 너무 많기 때문입니다. 따라서 최적의 파라미터 값으로 gamma를 10, cost를 1이라 가정한 채 프로젝트를 진행해 보았습니다.

```
tuned <- e1071::tune.svm(not.fully.paid ~ ., data = train, gamma = 10,cost = 1)

```
하지만 크게 의미가 없는 것이라 판단되었습니다.

따라서 데이터 양을 상당히 줄인 후 다시 진행해 보았습니다.(train test를 0.01:99.99 로 나눠 진행해 보았습니다.)

```
tuned <- e1071::tune.svm(not.fully.paid ~ ., data = train, gamma = 10^(-8:1),cost = 1:30)

summary(tuned)

best_param <- summary(tuned)$best.parameters
best_param
```
![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/05c4215d-83c4-447a-84fb-02fb62360fb2)


### 4-3. 하이퍼 파라미터를 적용한 svm모델 생성

train 데이터를 이용하여 위에서 구한 하이퍼 파라미터를 적용한 svm 모델을 생성하고 분류 결과를 확인해 보겠습니다.

confusionMatrix를 생성하여 분류결과의 정확도 및 성능지표를 확인해 보겠습니다.

```
svm_best <- e1071::svm(formula = not.fully.paid ~ ., data = train, type = "C-classification", kernel = "radial", gamma = best_param[1,1], cost = best_param[1,2])
summary(svm_best)

print("svm_best : train 데이터 분류 결과")
table(predict(svm_best, train), train$not.fully.paid)

print("svm_best : train 데이터 confusionMatrix 결과")
cm <- caret::confusionMatrix(predict(svm_best, train), train$not.fully.paid)
cm
```

![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/73809b19-83c7-407c-bf13-c386c4e146b6)

이제 test 데이터를 통해 모델의 예측력을 확인해 보겠습니다.

confusionMarix()를 생성하여 성능지표 역시 확인해 보겠습니다.

```
predict_value_svm <- predict(svm_best, test, type = "C-classification") %>%
  tibble(predict_value_svm = .)
predict_check_svm <-test %>% select(not.fully.paid) %>% dplyr::bind_cols(.,predict_value_svm)
head(predict_check_svm)

cm <- caret::confusionMatrix(predict(svm_best,test), test$not.fully.paid)
cm

```
![image](https://github.com/auspicious0/Loan_SVM/assets/108572025/2bf23f9b-1a29-4f42-93b7-b984b51cb7d3)


## 5. 결론

예측을 진행함에 있어서 하이퍼 파라미터로 변경 된 것과 또 train 을 통해 svm 벡터를 생성한 것과 원래 svm 벡터 사이의 차이가 존재하지 않았습니다.

이는 최적의 파라미터를 찾는 과정에서 넓은 변수의 스펙트럼을 통해 찾지 못한 것에 폐착 요인이 있을 수 있습니다.

따라서 제가 임의로 데이터를 줄여 넓은 변수 스펙트럼 하에 하이퍼 파라미터를 찾아 보았지만 데이터가 줄었다는 것에 의미가 손실된다고 생각합니다.

고로 이러한 문제를 해결할 수 있는 방법이나 제가 차마 놓친 개념이 있다면 [이슈 페이지](https://github.com/auspicious0/MovieRevenue/issues)를 통해 제출 부탁드립니다.

보다 더 자세한 내용을 원하신다면  [보고서](https://github.com/auspicious0/MovieRevenue/blob/main/boxoffice_RandomForest.ipynb) 를 확인해 주시기 바랍니다.



