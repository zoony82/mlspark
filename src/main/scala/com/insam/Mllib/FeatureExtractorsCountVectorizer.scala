package com.insam.Mllib

import org.apache.spark.sql.{ SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer}

object FeatureExtractorsCountVectorizer {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").getOrCreate()

    val testSeq = Seq(
      (0,Array("a","b","c")),
      (1,Array("a","b","a","c","b"))
    )
    val testDF = spark.createDataFrame(testSeq).toDF("id","words")
    testDF.show()

    val countVectorizerModel = new CountVectorizer().
      setInputCol("words").
      setOutputCol("features").
      setVocabSize(3).
      setMinDF(2).
      fit(testDF)

    val result = countVectorizerModel.transform(testDF)
    result.show()

    result.collect().foreach(v=>println(v))

  }
}
