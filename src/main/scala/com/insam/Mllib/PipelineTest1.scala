package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{Pipeline,PipelineModel}
import org.apache.spark.ml.feature.{HashingTF,Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row


object PipelineTest1 {

  val spark = SparkSession.builder().master("local").appName("pipeline test").getOrCreate()

  val trainData = spark.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0)
  )).toDF("id", "text", "label")

  trainData.show()

  // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr

  val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
  val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
  val logisticRegression = new LogisticRegression().setMaxIter(10).setRegParam(0.01)
  val pipeline = new Pipeline().setStages(Array(tokenizer,hashingTF,logisticRegression))

  // fit the pipeline to training documents
  val model = pipeline.fit(trainData)

  // Now we can optionally save the fitted pipeline to disk
  model.write.overwrite().save("/home/jjh/문서/dataset/modelResult")

  // We can also save this unfit pipeline to disk
  pipeline.write.overwrite().save("/home/jjh/문서/dataset/unfit-lr-model")

  // And load it back in during production
  val sameModel = PipelineModel.load("/home/jjh/문서/dataset/modelResult")

  // Prepare test documents, which are unlabeled (id, text) tuples.
  val testData = spark.createDataFrame(Seq(
    (4L, "spark i j k"),
    (5L, "l m n"),
    (6L, "spark hadoop spark"),
    (7L, "apache hadoop")
  )).toDF("id", "text")

  // Make predictions on test documents.
  val result: Array[Row] = model.transform(testData).
    select("id", "text", "probability", "prediction").
    collect()

  result.foreach{
    case Row(id:Long, text:String, prob:Vector, prediction:Double) =>
      println(s"($id,$text) => prob=$prob, predcition=$prediction")
  }
  
}
