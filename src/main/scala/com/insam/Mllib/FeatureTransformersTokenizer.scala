package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{Tokenizer,RegexTokenizer}
import org.apache.spark.sql.functions._

object FeatureTransformersTokenizer {
  //Tokenizer : taking text (such as a sentence) and breaking it into individual terms (usually words)

  val spark = SparkSession.builder().master("local").getOrCreate()

  val sentenceDataFrame = spark.createDataFrame(Seq(
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
  )).toDF("id", "sentence")

  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("word")
  val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("word").setPattern("\\W")

  val countToken = udf{ (v:Seq[String]) => v.length }

  import spark.implicits._

  val tokenized = tokenizer.transform(sentenceDataFrame)
  val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
  tokenized.withColumn("count",countToken($"word")).select("sentence","word","count").show(false)
  regexTokenized.withColumn("count",countToken($"word")).select("sentence","word","count").show(false)


}
