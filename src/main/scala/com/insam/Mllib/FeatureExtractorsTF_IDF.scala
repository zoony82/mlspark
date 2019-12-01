package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

object FeatureExtractorsTF_IDF {
  //tf-idf
  //tf : 문서에서 해당 단어가 얼마나 나왔는가
  //idf : 전체 문서들에서, 몇 개의 문서에 나타났는가(해당 단어가 나타난 문서 수/ 전체 문서 수)
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

    //https://stackoverflow.com/questions/35205865/what-is-the-difference-between-hashingtf-and-countvectorizer-in-spark/42200215
    /*
    partially reversible (CountVectorizer) vs irreversible (HashingTF) - since hashing is not reversible you cannot restore original input from a hash vector. From the other hand count vector with model (index) can be used to restore unordered input. As a consequence models created using hashed input can be much harder to interpret and monitor.
    memory and computational overhead - HashingTF requires only a single data scan and no additional memory beyond original input and vector. CountVectorizer requires additional scan over the data to build a model and additional memory to store vocabulary (index). In case of unigram language model it is usually not a problem but in case of higher n-grams it can be prohibitively expensive or not feasible.
    hashing depends on a size of the vector , hashing function and a document. Counting depends on a size of the vector, training corpus and a document.
    a source of the information loss - in case of HashingTF it is dimensionality reduction with possible collisions. CountVectorizer discards infrequent tokens.
    How it affects downstream models depends on a particular use case and data.
     */

  }
}
