package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

object RegressionLinear {
  // Linear Regression :
  // 종속 변수 y와, 한 개 이상의 독립 변수 X와의 선형 상관 관계를 모델링하는 회귀분석 기법
  // 한 개의 설명 변수에 기반한 경우에는 단순 선형 회귀, 둘 이상의 설명 변수에 기반한 경우에는 다중 선형 회귀

  val spark = SparkSession.builder().master("local").getOrCreate()
  val dataSet = spark.read.format("libsvm").load("/home/jjh/문서/dataset/spark_mllib/sample_linear_regression_data.txt")
  dataSet.show(false)
  dataSet.printSchema()

  val linearRegression = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

  var lrModel = linearRegression.fit(dataSet)

  // Print the coefficients and intercept for linear regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // Summarize the model over the training set and print out some metrics
  val trainingSummary = lrModel.summary
  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")
}
