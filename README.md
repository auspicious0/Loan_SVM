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

[Kaggle Dataset](#https://www.kaggle.com/datasets/sarahvch/predicting-who-pays-back-loans?select=loan_data.csv)


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
install.packages("Hmisc")
library(Hmisc)
mr$budget <- impute(mr$budget, median) #mean, median, 특정숫자

#mr$revenue <- impute(mr$revenue, median) #mean, median, 특정숫자
#아무래도 revenue 는 반응변수 종속변수이다 보니 같은 숫자가 너무 많으면 
#문제가 될 것으로 예상됩니다. 또한 남은 결측값이 많긴 하지만
#데이터가 너무 많아 후에 있을 분석 진행에 차질이 발생하여
#na.omit으로 삭제한 후 진행하겠습니다.
mr <- mr %>% na.omit()
```

이상값 처리를 진행해보겠습니다.

```
# 이상치 및 결측값 처리 함수

calculate_outliers <- function(data, column_name) {
  iqr_value <- IQR(data[[column_name]])
  upper_limit <- summary(data[[column_name]])[5] + 1.5 * iqr_value
  lower_limit <- summary(data[[column_name]])[2] - 1.5 * iqr_value

  data[[column_name]] <- ifelse(data[[column_name]] < lower_limit | data[[column_name]] > upper_limit, NA, data[[column_name]])

  return(data)
}
table(is.na(mr))
boxplot(mr$budget,mr$popularity,mr$runtime,mr$revenue)
# 이상치 및 결측값 처리 및 결과에 대한 상자그림 그리기
mr <- calculate_outliers(mr, "budget")
mr <- calculate_outliers(mr, "popularity")
mr <- calculate_outliers(mr, "runtime")
mr <- calculate_outliers(mr, "revenue")

table(is.na(mr))
mr <- na.omit(mr)
table(is.na(mr))
boxplot(mr$budget,mr$popularity,mr$runtime,mr$revenue)#char형 변수를 제외하고 정수형 변수만을  boxplot을 그려보겠습니다.

```
![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/60efe2ae-f0b9-4475-86e4-14c259a60e14)

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/6ef56431-3c9a-44f1-8a10-9b9fe3b9cdf7)

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/22d4dcd9-6210-40bd-9dd1-3d3571b0be44)

## 4. bagging 및 RandomForest 분석

데이터를 학습 및 테스트 세트로 분할하고 모델을 생성하여 성능을 평가하고 직관적 이해를 돕기 위해 시각화 해보겠습니다.

### 4-1. 회귀 분석

무작위로 데이터를 분리하지 않고 반응변수를 중심으로 8:2로 나누기 위해 caret::createDataPartition을 사용하겠습니다.

```
install.packages("caret")
library(caret)
index <- caret::createDataPartition(y = mr$revenue, p = 0.8, list = FALSE)
train <- mr[index, ]
test <- mr[-index, ]
```

우선 bagging 모델을 생성하고 bagging 모델의 예측력을 확인해 보겠습니다.

```
model_bagging <-ipred::bagging(revenue ~ ., data = train, nbagg = 100)
predict_value_bagging <- predict(model_bagging, test, type = "class")%>%
  tibble(predict_value_bagging = .)
predict_check_bagging <- test %>% select(revenue)%>%dplyr::bind_cols(.,predict_value_bagging)
predict_check_bagging
```


![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/80125650-8b87-4a4d-a11b-9db9f9fdf0e7)


예측값이 우수하지 않아 보입니다. 

따라서 회귀 분석은 여기서 종료하겠습니다.

### 4-2. 분류 분석

분류 분석을 위해 정수형 변수 revenue를 factor형으로 변환해야 합니다. 

이를 위해

revenue의 평균값 미만 데이터는 0,

revenue의 평균값 이상 데이터는 1로 변환 후

factor형으로 형변환하겠습니다.

```
# revenue 열의 평균값 계산
revenue_mean <- mean(mr$revenue, na.rm = TRUE)

# 변환: revenue 열 값이 revenue 평균값 미만인 경우 0, 이상인 경우 1로 변경
mr$revenue <- ifelse(mr$revenue < revenue_mean, 0, 1)

# factor 데이터 유형으로 변환
mr$revenue <- factor(mr$revenue)
```

모델을 사용하여 test 데이터로 예측을 수행한 후 예측값을 저장하고 실제 데이터와 대조하여 확인해 보겠습니다.
(앞 코드와 동일합니다.)

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/92413583-297a-4e99-b4bc-49e7cf34415d)

예측된 결과 어느 정도 예측을 수행한 것으로 확인할 수 있습니다. 

confusionMatrix를 활용하여 성능지표를 확인하겠습니다.

```
cm <- caret::confusionMatrix(predict_check_bagging$predict_value_bagging,test$revenue)
cm
```

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/b7b702e4-9943-45ee-9a6f-14567ba835fd)

이제 train 데이터로 RandomForest 모델을 만들어 보겠습니다.

```
library(randomForest)
model_rf <- randomForest(revenue ~ ., data = train, na.action = na.omit, importance = T, mtry = 7, ntree = 1000)
model_rf
```

만든 랜덤포레스트 모델로 예측을 수행한 후 실제 값과 결과를 비교해 보겠습니다.

(앞 코드와 동일합니다.)

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/e7e29874-e987-4a85-b032-288c381130b1)

예측을 잘 수행한 것을 확인할 수 있습니다.

이제 예측값을 저장한 데이터와 실제 데이터 사이의 confusionMatrix를 생성한 후 성능지표를 확인해 보겠습니다.

```
cm <- caret::confusionMatrix(predict_check_rf$predict_value_rf,test$revenue)
cm
```

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/555aa85a-9a96-4fe6-981c-69677ef9c7ff)


RandomForest 보다 bagging이 정확도, 민감도 등 여러 측면에서 나은 결과를 보이는 것을 확인할 수 있습니다. (75프로)

모델에서 변수의 중요도를 그림으로 나타내 보겠습니다.

```
varImpPlot(model_rf, type = 2, pch = 19, col = 1, cex = 1, main = "")
```

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/6075d09e-17a6-4b59-aa30-3e565217c99a)

수익(revenue)에 가장 중요한 요소는 budget(예산)과 인기, 상영시간, 장르 순으로 이루어진 것을 확인할 수 있습니다.


## 5. 문의
프로젝트에 관한 문의나 버그 리포트는 [이슈 페이지](https://github.com/auspicious0/MovieRevenue/issues)를 통해 제출해주세요.

보다 더 자세한 내용을 원하신다면 [보고서](https://github.com/auspicious0/MovieRevenue/blob/main/boxoffice_RandomForest.ipynb) 를 확인해 주시기 바랍니다.
