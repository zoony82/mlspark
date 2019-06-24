package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

object FeatureExtractorsTF_IDF {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().master("local").getOrCreate()
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat"),
      (1.0, "spark"),
      (1.0, "spark"),
      (1.0, "spark"),
      (1.0, "english")
    )).toDF("label","sentence")

    sentenceData.show()

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    wordsData.show()

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(50)
    val featurezedData = hashingTF.transform(wordsData)
    featurezedData.show()

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurezedData)

    val reScaleData = idfModel.transform(featurezedData)
    reScaleData.show()
    reScaleData.select("words","rawFeatures","features").toJSON.foreach(println(_))
    // spark 문자열을 추가할 수록 english 의 tf-idf값이 올라감

  }
}
