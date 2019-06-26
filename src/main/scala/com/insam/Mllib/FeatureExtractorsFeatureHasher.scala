package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.FeatureHasher

object FeatureExtractorsFeatureHasher {
  //Feature Hasher : 임의의 feature 들을 vector 안의 index 로 바꿔줌
  // a set of categorical or numerical features into a feature vector of specified dimension
  val spark = SparkSession.builder().master("local").getOrCreate()
  val dataset = spark.createDataFrame(Seq(
    (2.2,true,"1","foo"),
    (3.3, false, "2", "bar"),
    (4.4, false, "3", "baz"),
    (5.5, false, "4", "foo")
  )).toDF("real","bool","stringNum","string")

  val hasher = new FeatureHasher().setInputCols("real","bool","stringNum","string").setOutputCol("features")
  val featured = hasher.transform(dataset)
  featured.show()

  featured.show(false)

}
