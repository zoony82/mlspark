package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object ClusteringKmeans {
  // K-means
  // 주어진 데이터를 k개의 클러스터로 묶는 알고리즘
  // 각 클러스터와 거리 차이의 분산을 최소화하는 방식으로 동작
  val spark = SparkSession.builder().master("local").getOrCreate()
  val dataset = spark.read.format("libsvm").load("/home/jjh/문서/dataset/spark_mllib/sample_kmeans_data.txt")

  dataset.show(false)

  // train the model
  val kmeans = new KMeans().setK(2).setSeed(1L)
  val model = kmeans.fit(dataset)


  // make prediction
  val prediction = model.transform(dataset)
  prediction.show()

  // evaluate clustering by computing Silhouette score
  val evaluator = new ClusteringEvaluator()
  val evaluResult = evaluator.evaluate(prediction)
  println(s"evaluter is $evaluResult")

  // show result
  println("Cluster center is ")
  model.clusterCenters.foreach(println)

}
