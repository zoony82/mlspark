package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors

object FeatureTransformersNormalizer {
  // Normalizer : dataset of Vector rows, normalizing each Vector to have unit norm
  val spark = SparkSession.builder().master("local").getOrCreate()

  val dataFrame = spark.createDataFrame(Seq(
    (0, Vectors.dense(1.0, 0.5, -1.0)),
    (1, Vectors.dense(2.0, 1.0, 1.0)),
    (2, Vectors.dense(4.0, 10.0, 2.0))
  )).toDF("id", "features")

  val normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(1.0)

  val l1NormData = normalizer.transform(dataFrame)
  println("Normalized using L^1 norm")
  l1NormData.show()

  val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
  println("Normalized using L^inf norm")
  lInfNormData.show()

}
