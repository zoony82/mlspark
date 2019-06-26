package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vector

object FeatureExtractorsW2V {
  //W2V : 문자를 특정 차원에서의 위치를 부여함
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").getOrCreate()

    val docSeq: Seq[Array[String]] = Seq(
      "Hi I heard about Spark".split(" "),
      "sweet apple".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" "),
      "sweet apple".split(" ")
    )
//    docSeq.foreach(v=>println(v(0) +" " + v(1)))

    val docSeq_ : Seq[Tuple1[Array[String]]] = docSeq.map(Tuple1.apply) //https://z-cube.tistory.com/6
//    docSeq_.foreach(v => println(v._1(0) +" " + v._1(1)))
    val docDF = spark.createDataFrame(docSeq_).toDF("text")
//    docDF.show()
//    docDF.printSchema()

    // Learn a mapping from words to Vectors
    val word2Vec = new Word2Vec().
      setInputCol("text").
      setOutputCol("result").
      setVectorSize(3).
      setMinCount(0)

    val model = word2Vec.fit(docDF)

    val result = model.transform(docDF)
    result.show()

    // sweet apple는 3차원상 같은곳에 존재
    result.collect().foreach { case Row(text: Seq[_], features: Vector) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }

  }
}
