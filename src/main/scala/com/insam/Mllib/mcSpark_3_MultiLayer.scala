package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.ml.Pipeline

object mcSpark_3_MultiLayer {
  // classification
  // 멀티 레이어를 파이프라인을 통해 학습
  val spark = SparkSession.builder().master("local").getOrCreate()
  val data = spark.read.format("org.apache.spark.csv").option("header", true).option("inferSchema", "true").csv("D:\\ml\\letterdata.data")
  data.show(5)
  data.printSchema()

  // 숫자로 되어있는 속서읃ㄹ을 다 모아서 핏쳐로 만들기
  val assembler = new VectorAssembler().setInputCols(Array("xbox", "ybox", "width", "height", "onpix","xbar","ybar","x2bar","y2bar","xybar","x2ybar","xy2bar", "xedge","xedgey","yedge","yedgex")).setOutputCol("features")
  // 또 다른 dataframe
  val assembler_res = assembler.transform(data)
  assembler_res.show(5,false)

  // Index labels, adding metadata to the label column.
  // Fit on whole dataset to include all labels in index.
  val labelIndexer = new StringIndexer().setInputCol("letter").setOutputCol("indexedLabel").fit(assembler_res)
  println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")


  // Automatically identify categorical features, and index them.
  // Set maxCategories so features with > 4 distinct values are treated as continuous.
  // 총 feature가 16개 이기 때문에...
  val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(16).fit(assembler_res)
  println(featureIndexer) // 모델 형태로 나왔음

  val splits = assembler_res.randomSplit(Array(0.6, 0.4))
  val trainingData = splits(0)
  val testData = splits(1)
  // 그런데 중간에 dataset로 바뀐다....Dataset[org.apache.spark.sql.Row]


  val layers = Array[Int](16,48,48,26)
  // create the trainer and set its parameters
  val trainer = new MultilayerPerceptronClassifier().
    setLayers(layers).
    setLabelCol("indexedLabel").
    setFeaturesCol("indexedFeatures").
    setBlockSize(128).
    setSeed(System.currentTimeMillis).
    setMaxIter(400)

  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString().
    setInputCol("prediction").
    setOutputCol("predictedLabel").
    setLabels(labelIndexer.labels)

  // Chain indexers and MultilayerPerceptronClassifier in a Pipeline.
  val pipeline = new Pipeline().
    setStages(Array(labelIndexer, featureIndexer, trainer, labelConverter))

  // Train model. This also runs the indexers.
  // 에스터메이터가 등장하고, 데이터프레임을 통해 훈련한다.
  val model = pipeline.fit(trainingData)

  // Make predictions.
  // test 데이터는 앞의 핏쳐 엔지니어링 과정을 안거쳐도, 알아서 파이프라인에서 진행 됨
  // 한번 나온 모델을 사용할때는 transform을 쓰는거다.
  val predictions = model.transform(testData)


  // Select example rows to display.
  predictions.select("letter","predictedLabel").show(100)
  import spark.implicits._

  //true 7,045
  predictions.select("letter","predictedLabel").filter($"letter"===$"predictedLabel").count()

  //false 963
  predictions.select("letter","predictedLabel").filter($"letter"=!=$"predictedLabel").count()

  // Select (prediction, true label) and compute test error.
  val evaluator = new MulticlassClassificationEvaluator().
    setLabelCol("indexedLabel").
    setPredictionCol("prediction").
    setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println("Test Error = " + (1.0 - accuracy))

  val mlpc = model.stages(2).asInstanceOf[MultilayerPerceptronClassificationModel]

  println(s"Learned classification model:\n$mlpc")
  println(s"Params: ${mlpc.explainParams}")
  println(s"Weights: ${mlpc.weights}")

}
