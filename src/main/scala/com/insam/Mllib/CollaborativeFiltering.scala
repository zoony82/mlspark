package com.insam.Mllib

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator

object CollaborativeFiltering {
  //  사용자들로부터 얻은 기호정보에 따라 사용자들의 관심사들을 자동적으로 예측하게 해주는 방법
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").getOrCreate()

    val ratings = spark.read.textFile("/home/jjh/문서/dataset/spark_mllib/als/sample_movielens_ratings.txt")
    ratings.show(100)

    import spark.implicits._
    val ratingParse = ratings.map(v=>parseRating(v)).toDF()
    ratingParse.show(false)
    ratingParse.printSchema()

    val Array(training, test) = ratingParse.randomSplit(Array(0.8,0.2))

    training.count()
    test.count()

    //build the recommendation model, using ALS on the training data
    val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").
      setRatingCol("rating")
    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    model.setColdStartStrategy("drop")
    val prediction = model.transform(test)
    prediction.show()

    val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating").setPredictionCol("prediction")

    val rmse = evaluator.evaluate(prediction)
    println(rmse)

    // Generate top 10 movie recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    userRecs.show()
    // Generate top 10 user recommendations for each movie
    val movieRecs = model.recommendForAllItems(10)

    // Generate top 10 movie recommendations for a specified set of users
    val users = ratings.select(als.getUserCol).distinct().limit(3)
    val userSubsetRecs = model.recommendForUserSubset(users, 10)
    // Generate top 10 user recommendations for a specified set of movies
    val movies = ratings.select(als.getItemCol).distinct().limit(3)
    val movieSubSetRecs = model.recommendForItemSubset(movies, 10)

  }

  case class Rating(userId:Int, movieId:Int, rating:Float, timestamp:Long)

  def parseRating(str:String):Rating={
    val fields = str.split("::")
    assert(fields.size==4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

}
