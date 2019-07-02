package com.insam.Mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object mcSpark_2 {
  /*
  yarn을 통해 프로세스 제거 가능
  yarn application -kill application_1561948442730_0008
   */

  val conf:SparkConf = new SparkConf().setAppName("Histogram").setMaster("local")
  val sc= new SparkContext(conf)

  // classfication

  // sed 1d train.tsv > train_noheader.tsv
  // 데이터 적재
  val rawData = sc.textFile("D:\\classfication\\train_noheader\\train_noheader.tsv")
  val records = rawData.map(line => line.split("\t"))
  records.first



  val data = records.map { r =>
    val trimmed = r.map(_.replaceAll("\"", ""))
    val label = trimmed(r.size - 1).toInt
    val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
    LabeledPoint(label, Vectors.dense(features))
  }
  data.cache
  val numData = data.count
  data.first()

  // 나이브 베이즈를 위해 음수로 되어 있는 데이터는 0으로 변환
  val nbData = records.map { r =>
    val trimmed = r.map(_.replaceAll("\"", ""))
    val label = trimmed(r.size - 1).toInt
    val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
    LabeledPoint(label, Vectors.dense(features))
  }

  nbData.first()

  //  Logistic Regression model 데이터 훈련
  import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
  // 스토캐스틱 그래디언트 디센트 알고리즘 - 확률 이용
  // 런닝레이트를 조정 해주는거?

  import org.apache.spark.mllib.classification.SVMWithSGD
  import org.apache.spark.mllib.classification.NaiveBayes
  import org.apache.spark.mllib.tree.DecisionTree
  import org.apache.spark.mllib.tree.configuration.Algo
  import org.apache.spark.mllib.tree.impurity.Entropy

  val numIterations = 10
  val maxTreeDepth = 5
  val lrModel = LogisticRegressionWithSGD.train(data, numIterations) // 모델1
  val svmModel = SVMWithSGD.train(data, numIterations) // 모델 2

  // 다른 모델 학습
  val nbModel = NaiveBayes.train(nbData) // 모델 3
  val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth) // 모델 4


  // 데이터 예측
  val dataPoint = data.first
  val prediction = lrModel.predict(dataPoint.features) // 예측한 값
  val trueLabel = dataPoint.label // 실제 값

  val predictions = lrModel.predict(data.map(lp => lp.features))





  // logistic regression 정확도 계산,  Double = 3806.0
  val lrTotalCorrect = data.map { point =>
    if (lrModel.predict(point.features) == point.label) 1 else 0
  }.sum


  // (정확하게 분류된 수 / 전체 데이터 수) 0.51
  val lrAccuracy = lrTotalCorrect / numData


  // 다른 모델의 정확도 계산
  val svmTotalCorrect = data.map { point =>
    if (svmModel.predict(point.features) == point.label) 1 else 0
  }.sum
  val nbTotalCorrect = nbData.map { point =>
    if (nbModel.predict(point.features) == point.label) 1 else 0
  }.sum
  // decision tree 는 임계치 설정이 필요
  val dtTotalCorrect = data.map { point =>
    val score = dtModel.predict(point.features)
    val predicted = if (score > 0.5) 1 else 0
    if (predicted == point.label) 1 else 0
  }.sum


  // 각 모델의 정확도
  val svmAccuracy = svmTotalCorrect / numData
  val nbAccuracy = nbTotalCorrect / numData
  val dtAccuracy = dtTotalCorrect / numData


  // Precision-Recall 과  ROC curves 값을 계산
  import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
  val metrics = Seq(lrModel, svmModel).map { model =>
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
  }
  // naive Bayes metrics
  val nbMetrics = Seq(nbModel).map{ model =>
    val scoreAndLabels = nbData.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
  }

  // 분할 트리의 PR, ROC 계산
  val dtMetrics = Seq(dtModel).map{ model =>
    val scoreAndLabels = data.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
  }
  val allMetrics = metrics ++ nbMetrics ++ dtMetrics
  allMetrics.foreach{ case (m, pr, roc) =>
    println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%")
  }
  /*
  LogisticRegressionModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
  SVMModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
  NaiveBayesModel, Area under PR: 68.0851%, Area under ROC: 58.3559%
  DecisionTreeModel, Area under PR: 74.3081%, Area under ROC: 64.8837%
  */

  // 어떻게 하면 성능을 높힐 수 있을까?
  // 회귀를 더 제대로 하려면 숫자를 정규화
  // 나이브(확률)을 더 제대로 하려먼 ?


  // 성능 개선
  // 숫 자 데이터 표준화
  import org.apache.spark.mllib.linalg.distributed.RowMatrix
  val vectors = data.map(lp => lp.features)
  val matrix = new RowMatrix(vectors)
  val matrixSummary = matrix.computeColumnSummaryStatistics()

  println(matrixSummary.mean)
  // [0.41225805299526636,2.761823191986623,0.46823047328614004, ...
  println(matrixSummary.min)
  // [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.045564223,-1.0, ...
  println(matrixSummary.max)
  // [0.999426,363.0,1.0,1.0,0.980392157,0.980392157,21.0,0.25,0.0,0.444444444, ...
  println(matrixSummary.variance)
  // [0.1097424416755897,74.30082476809638,0.04126316989120246, ...
  println(matrixSummary.numNonzeros)
  // [5053.0,7354.0,7172.0,6821.0,6160.0,5128.0,7350.0,1257.0,0.0,7362.0, ...


  // MLlib's StandardScaler 사용하여 데이터 표준화
  import org.apache.spark.mllib.feature.StandardScaler
  val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors) // z score 표준화
  val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
  // 표준화 된 데이터와 원래 데이터 비교
  println(data.first.features)
  // [0.789131,2.055555556,0.676470588,0.205882353,
  println(scaledData.first.features) // 데이터가 정규성을 띄고 있을때만 사용할 수 있는 방법
  // [1.1376439023494747,-0.08193556218743517,1.025134766284205,-0.0558631837375738,
  println((0.789131 - 0.41225805299526636)/math.sqrt(0.1097424416755897))
  // 1.137647336497682



  // 표준화된 데이터로  logistic regression model 학습
  // logistic regression 은 데이터를 정규화 하니깐 확률이 높아지네...(z score 표준화)
  val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
  val lrTotalCorrectScaled = scaledData.map { point =>
    if (lrModelScaled.predict(point.features) == point.label) 1 else 0
  }.sum
  val lrAccuracyScaled = lrTotalCorrectScaled / numData
  // lrAccuracyScaled: Double = 0.6204192021636241
  val lrPredictionsVsTrue = scaledData.map { point =>
    (lrModelScaled.predict(point.features), point.label)
  }
  val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
  val lrPr = lrMetricsScaled.areaUnderPR
  val lrRoc = lrMetricsScaled.areaUnderROC
  println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%")
  /*
  LogisticRegressionModel
  Accuracy: 62.0419%
  Area under PR: 72.7254%
  Area under ROC: 61.9663% => 좀 높아졌네
  */



  //  'category' feature 추가
  val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap // zipWithIndex 각 값마다 고유한 값으로 인덱스
  // categories: scala.collection.immutable.Map[String,Int] = Map("weather" -> 0, "sports" -> 6,
  //	"unknown" -> 4, "computer_internet" -> 12, "?" -> 11, "culture_politics" -> 3, "religion" -> 8,
  // "recreation" -> 2, "arts_entertainment" -> 9, "health" -> 5, "law_crime" -> 10, "gaming" -> 13,
  // "business" -> 1, "science_technology" -> 7)
  val numCategories = categories.size
  // numCategories: Int = 14
  val dataCategories = records.map { r =>
    val trimmed = r.map(_.replaceAll("\"", ""))
    val label = trimmed(r.size - 1).toInt
    val categoryIdx = categories(r(3))
    val categoryFeatures = Array.ofDim[Double](numCategories)
    categoryFeatures(categoryIdx) = 1.0
    val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
    val features = categoryFeatures ++ otherFeatures
    LabeledPoint(label, Vectors.dense(features))
  }
  // LabeledPoint(0.0, [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,
  //	0.676470588,0.205882353,0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,
  // 0.003883495,1.0,1.0,24.0,0.0,5424.0,170.0,8.0,0.152941176,0.079129575])


  // feature vectors 표준화
  val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
  val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))
  println(dataCategories.first.features)
  // [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,0.676470588,0.205882353,
  // 0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,0.003883495,1.0,1.0,24.0,0.0,
  // 5424.0,170.0,8.0,0.152941176,0.079129575]
  println(scaledDataCats.first.features)
  /*
  [-0.023261105535492967,2.720728254208072,-0.4464200056407091,-0.2205258360869135,-0.028492999745483565,
  -0.2709979963915644,-0.23272692307249684,-0.20165301179556835,-0.09914890962355712,-0.381812077600508,
  -0.06487656833429316,-0.6807513271391559,-0.2041811690290381,-0.10189368073492189,1.1376439023494747,
  -0.08193556218743517,1.0251347662842047,-0.0558631837375738,-0.4688883677664047,-0.35430044806743044
  ,-0.3175351615705111,0.3384496941616097,0.0,0.8288021759842215,-0.14726792180045598,0.22963544844991393,
  -0.14162589530918376,0.7902364255801262,0.7171932152231301,-0.29799680188379124,-0.20346153667348232,
  -0.03296720969318916,-0.0487811294839849,0.9400696843533806,-0.10869789547344721,-0.2788172632659348]
  */

  // 표준화된 데이터 학습 및 지표 계산
  val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
  val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
    if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
  }.sum
  val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
  val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
    (lrModelScaledCats.predict(point.features), point.label)
  }
  val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
  val lrPrCats = lrMetricsScaledCats.areaUnderPR
  val lrRocCats = lrMetricsScaledCats.areaUnderROC
  println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%")
  /*
  LogisticRegressionModel
  Accuracy: 66.5720%
  Area under PR: 75.7964%
  Area under ROC: 66.5483%
  */

  // 로지스틱회귀의 코스트 펑션은 시그모이드이고, 딥러닝에서 사용하니 중요하다?
  // 그런데 더 성능을 높히려면 로그를 적용해라?



  // train naive Bayes model with only categorical data
  // 확률가지고 데이터를 나눌껀데, 조건부 확률을 사용할꺼다.

  val dataNB = records.map { r =>
    val trimmed = r.map(_.replaceAll("\"", ""))
    val label = trimmed(r.size - 1).toInt
    val categoryIdx = categories(r(3))
    val categoryFeatures = Array.ofDim[Double](numCategories)
    categoryFeatures(categoryIdx) = 1.0
    LabeledPoint(label, Vectors.dense(categoryFeatures)) // 범주형 데이터만 활용 했다.
  }

  val nbModelCats = NaiveBayes.train(dataNB)
  val nbTotalCorrectCats = dataNB.map { point =>
    if (nbModelCats.predict(point.features) == point.label) 1 else 0
  }.sum
  val nbAccuracyCats = nbTotalCorrectCats / numData
  val nbPredictionsVsTrueCats = dataNB.map { point =>
    (nbModelCats.predict(point.features), point.label)
  }
  val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
  val nbPrCats = nbMetricsCats.areaUnderPR
  val nbRocCats = nbMetricsCats.areaUnderROC
  println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy: ${nbAccuracyCats * 100}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%")
  /*
  NaiveBayesModel
  Accuracy: 60.9601%
  Area under PR: 74.0522%
  Area under ROC: 60.5138%
  */


  // investigate the impact of model parameters on performance
  // 파라메터를 평가해보자
  // create a training function
  import org.apache.spark.rdd.RDD
  import org.apache.spark.mllib.optimization.Updater
  import org.apache.spark.mllib.optimization.SimpleUpdater
  import org.apache.spark.mllib.optimization.L1Updater
  import org.apache.spark.mllib.optimization.SquaredL2Updater
  import org.apache.spark.mllib.classification.ClassificationModel

  // helper function to train a logistic regresson model
  def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = { // stepSize : 러닝레이트
    val lr = new LogisticRegressionWithSGD
    lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
    lr.run(input)
  }
  // helper function to create AUC metric
  def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
    val scoreAndLabels = data.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (label, metrics.areaUnderROC)
  }


  // cache the data to increase speed of multiple runs agains the dataset
  scaledDataCats.cache  // 가능하면 lru에 의해서 밀어나지 않도록 cache 사용
  // num iterations
  val iterResults = Seq(1, 5, 10, 50).map { param =>
    val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
    createMetrics(s"$param iterations", scaledDataCats, model)
  }
  iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
  /*
  1 iterations, AUC = 64.97%
  5 iterations, AUC = 66.62%
  10 iterations, AUC = 66.55%
  50 iterations, AUC = 66.81%
  */


  // step size : 얼마만큼 러닝 레이트를 이동 시킬것인가
  val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
    val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
    createMetrics(s"$param step size", scaledDataCats, model)
  }
  stepResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
  /*
  0.001 step size, AUC = 64.95%
  0.01 step size, AUC = 65.00%
  0.1 step size, AUC = 65.52%
  1.0 step size, AUC = 66.55%
  10.0 step size, AUC = 61.92%
  */


  // regularization
  val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
    val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
    createMetrics(s"$param L2 regularization parameter", scaledDataCats, model) // 제곱값으로 업데이트 ->SquaredL2Updater
  }
  regResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
  /*
  0.001 L2 regularization parameter, AUC = 66.55%
  0.01 L2 regularization parameter, AUC = 66.55%
  0.1 L2 regularization parameter, AUC = 66.63%
  1.0 L2 regularization parameter, AUC = 66.04%
  10.0 L2 regularization parameter, AUC = 35.33%
  */


  // decision tree
  import org.apache.spark.mllib.tree.impurity.Impurity //
  import org.apache.spark.mllib.tree.impurity.Entropy //
  import org.apache.spark.mllib.tree.impurity.Gini //
  def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
    DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
  }

  //  tree depth 에 따른 Entropy impurity
  val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
    val model = trainDTWithParams(data, param, Entropy)
    val scoreAndLabels = data.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$param tree depth", metrics.areaUnderROC)
  }
  dtResultsEntropy.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
  /*
  1 tree depth, AUC = 59.33%
  2 tree depth, AUC = 61.68%
  3 tree depth, AUC = 62.61%
  4 tree depth, AUC = 63.63%
  5 tree depth, AUC = 64.88%
  10 tree depth, AUC = 76.26%
  20 tree depth, AUC = 98.45%
  */


  // 어떤 변수에 영햐을 많이 받았는지 확인해보자.
  data.first()
  val dtModel_ = DecisionTree.train(data, Algo.Classification, Entropy, 20)
  dtModel_.toDebugString
  // 얼마나 맞췄는지도 중요하지만, 얼마나 사람이 이해했는가도 중요하다.
  // 그리고 roc/auc도 중요하다.


  //  tree depth 에 따른 Gini impurity 영향평가
  val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
    val model = trainDTWithParams(data, param, Gini)
    val scoreAndLabels = data.map { point =>
      val score = model.predict(point.features)
      (if (score > 0.5) 1.0 else 0.0, point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$param tree depth", metrics.areaUnderROC)
  }
  dtResultsGini.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
  /*
  1 tree depth, AUC = 59.33%
  2 tree depth, AUC = 61.68%
  3 tree depth, AUC = 62.61%
  4 tree depth, AUC = 63.63%
  5 tree depth, AUC = 64.89%
  10 tree depth, AUC = 78.37%
  20 tree depth, AUC = 98.87%
  */


  // investigate Naive Bayes parameters
  def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
    val nb = new NaiveBayes
    nb.setLambda(lambda)
    nb.run(input)
  }
  val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
    val model = trainNBWithParams(dataNB, param)
    val scoreAndLabels = dataNB.map { point =>
      (model.predict(point.features), point.label)
    }
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    (s"$param lambda", metrics.areaUnderROC)
  }
  nbResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
  /*
  0.001 lambda, AUC = 60.51%
  0.01 lambda, AUC = 60.51%
  0.1 lambda, AUC = 60.51%
  1.0 lambda, AUC = 60.51%
  10.0 lambda, AUC = 60.51%
  */


  // illustrate cross-validation
  // create a 60% / 40% train/test data split
  val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
  val train = trainTestSplit(0)
  val test = trainTestSplit(1)
  // now we train our model using the 'train' dataset, and compute predictions on unseen 'test' data
  // in addition, we will evaluate the differing performance of regularization on training and test datasets
  val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
    val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
    createMetrics(s"$param L2 regularization parameter", test, model)
  }
  regResultsTest.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
  /*
  0.0 L2 regularization parameter, AUC = 66.480874%
  0.001 L2 regularization parameter, AUC = 66.480874%
  0.0025 L2 regularization parameter, AUC = 66.515027%
  0.005 L2 regularization parameter, AUC = 66.515027%
  0.01 L2 regularization parameter, AUC = 66.549180%
  */


  // training set results
  val regResultsTrain = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
    val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
    createMetrics(s"$param L2 regularization parameter", train, model)
  }
  regResultsTrain.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
  /*
  0.0 L2 regularization parameter, AUC = 66.260311%
  0.001 L2 regularization parameter, AUC = 66.260311%
  0.0025 L2 regularization parameter, AUC = 66.260311%
  0.005 L2 regularization parameter, AUC = 66.238294%
  0.01 L2 regularization parameter, AUC = 66.238294%
  */


}
