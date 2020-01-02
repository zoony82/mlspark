package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object FeatureExtractorsCountVectorizer {
  // CountVector : 문서를 token count matrix로 변환
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

    //count the frequency of words
    import spark.implicits._
    val result_cnt = result.withColumn("words_explode",explode($"words")).
      groupBy($"words_explode").
      agg(count($"words_explode").as("counts"))

    result_cnt.show()

    val result_id = result_cnt.
      withColumn("id",row_number().over(Window.orderBy("words_explode")) -1)

    result_id.show()

    //
    countVectorizerModel.vocabulary


  }
}
