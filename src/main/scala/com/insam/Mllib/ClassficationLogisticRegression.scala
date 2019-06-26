package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression

object ClassficationLogisticRegression {
  // Binomial logistic regression
  // 로지스틱 회귀의 목적은 일반적인 회귀 분석의 목표와 동일하게 종속 변수와 독립 변수간의 관계를 구체적인 함수로 나타내어 향후 예측 모델에 사용하는 것이다.
  // 이는 독립 변수의 선형 결합으로 종속 변수를 설명한다는 관점에서는 선형 회귀 분석과 유사하다.
  // 하지만 로지스틱 회귀는 선형 회귀 분석과는 다르게 종속 변수가 범주형 데이터를 대상으로 하며
  // 입력 데이터가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나뉘기 때문에 일종의 분류 (classification) 기법으로도 볼 수 있다.

  val spark = SparkSession.builder().master("local").getOrCreate()

  val training = spark.read.format("libsvm").load("/home/jjh/문서/dataset/spark_mllib/sample_libsvm_data.txt")

  training.show(false)
  training.printSchema()

  val logisticRegression = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
  val lrModel = logisticRegression.fit(training)

  // Print the coefficients and intercept for logistic regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // We can also use the multinomial family for binary classification
  val mlr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFamily("multinomial")

  val mlrModel = mlr.fit(training)

  // Print the coefficients and intercepts for logistic regression with multinomial family
  println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
  println(s"Multinomial intercepts: ${mlrModel.interceptVector}")


}
