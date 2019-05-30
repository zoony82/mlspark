package com.insam.Test

//https://github.com/dosht/spark-word2vec-example/blob/master/src/main/scala/Word2VecExample.scala

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.util.Try

case class Sample(id:String, review:String, sentiment:Option[Int] = None)

object Word2vec extends App{
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  def printRDD(xs: RDD[_]): Unit ={
    println("-------------------")
    xs take 5 foreach println
    println("-------------------")
  }

  val conf = new SparkConf(false).setMaster("local").setAppName("word2vec")
  val sc = new SparkContext(conf)

  // Load
  /*
  train = pd.read_csv('/home/insam/09_data/bagofwords/labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
  test = pd.read_csv('/home/insam/09_data/bagofwords/testData.tsv', header=0, delimiter="\t", quoting=3)
  unlabeled_train = pd.read_csv('/home/insam/09_data/bagofwords/unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)

   */
  val trainPath = s"/home/insam/09_data/bagofwords/labeledTrainData.tsv"
  val testPath = s"/home/insam/09_data/bagofwords/testData.tsv"

  // Load text
  def skipHeaders(idx:Int, iter:Iterator[String]) = if(idx == 0) iter.drop(1) else iter

  val trainFile = sc.textFile(trainPath) mapPartitionsWithIndex skipHeaders map (t => t.split("\t"))
  val testFile = sc.textFile(testPath) mapPartitionsWithIndex skipHeaders map (t => t.split("\t"))


  // To sample
  def toSample(segments: Array[String]) = segments match{
    case Array(id, sentiment, review) => Sample(id, review, Some(sentiment.toInt))
    case Array(id,review) => Sample(id,review)
  }

  val trainSamples = trainFile map toSample
  val testSamples = testFile map toSample

  trainSamples.first()

  // Clean Html
  def cleanHtml(str:String) = str.replaceAll("""<(?!\/?a(?=>|\s.*>))\/?.*?>""","")

  def cleanSampleHtml(sample:Sample) = sample copy (review = cleanHtml(sample.review))

  val cleanTrainSamples = trainSamples map cleanSampleHtml
  val cleanTestSamples = testSamples map cleanSampleHtml

  cleanTrainSamples.first()

  // Words only
  def cleanWord(str:String) = str.split(" ").map(_.trim.toLowerCase).filter(_.size > 0).map(_.replaceAll("\\W","")).reduce((x,y) => s"$x $y")

  def wordOnlySample(sample:Sample) = sample copy (review = cleanWord(sample.review))

  val wordOnlyTrainSample = cleanTrainSamples map wordOnlySample
  val wordOnlyTestSample = cleanTestSamples map wordOnlySample

  wordOnlyTrainSample.first()
  wordOnlyTrainSample.count() //25000

  wordOnlyTestSample.first()

  // Word2Vec
  val samplePairs = wordOnlyTrainSample.map(s => s.id -> s).cache()
  val reviewWordsPairs: RDD[(String, Iterable[String])] = samplePairs.mapValues(_.review.split(" ").toIterable)
  println("Start Training Word2Vec --->")

  reviewWordsPairs.first()
  reviewWordsPairs.count() // 25000
  val word2vecModel = new Word2Vec().fit(reviewWordsPairs.values)

  word2vecModel.transform("london")
  word2vecModel.findSynonyms("london",4)

  def wordFeatures(words: Iterable[String]): Iterable[Vector] = words.map(w => Try(word2vecModel.transform(w))).filter(_.isSuccess).map(_.get)

  def avgWordFeatures(wordFeatures: Iterable[Vector]): Vector = Vectors.fromBreeze(wordFeatures.map(_.toBreeze).reduceLeft(_ + _) / wordFeatures.size.toDouble)

  // Create a feature vectors
  val wordFeaturePair = reviewWordsPairs mapValues wordFeatures
  val avgWordFeaturesPair = wordFeaturePair mapValues avgWordFeatures
  val featuresPair = avgWordFeaturesPair join samplePairs mapValues {
    case (features, Sample(id, review, sentiment)) => LabeledPoint(sentiment.get.toDouble, features)
  }
  val trainingSet = featuresPair.values

  // Classification
  println("String Learning and evaluating models")
  val Array(x_train, x_test) = trainingSet.randomSplit(Array(0.7, 0.3))
  val model = SVMWithSGD.train(x_train, 100)

  val result = model.predict(x_test.map(_.features))

  println(s"10 samples:")
  x_test.map { case LabeledPoint(label, features) => s"$label -> ${model.predict(features)}" } take 10 foreach println
  val accuracy = x_test.filter(x => x.label == model.predict(x.features)).count.toFloat / x_test.count
  println(s"Model Accuracy: $accuracy")

  println("<---- done")
  Thread.sleep(10000)




































































}
