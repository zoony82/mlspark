package com.insam.Mllib

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.jblas.DoubleMatrix

object mcSpark_5_Recommand {
/*
http://jblas.org/download.html 추가 필요
제플린 설정

 */
  def main(args: Array[String]): Unit = {
    /*
    Ensure you start the shell with sufficient memory: ./bin/spark-shell --driver-memory 4g
  */

    val conf:SparkConf = new SparkConf().setAppName("Histogram").setMaster("local")
    val sc= new SparkContext(conf)

    /* Load the raw ratings data */
    val rawData = sc.textFile("D:\\ml-100k\\u.data")
    rawData.first()
    // res24: String = 196	242	3	881250949(userid,movieid,rating,timestamp)


    /* Extract the user id, movie id and rating  */
    val rawRatings = rawData.map(_.split("\t").take(3))
    rawRatings.first()
    // res25: Array[String] = Array(196, 242, 3)

    /* Construct the RDD of Rating objects */
    val ratings = rawRatings.map { case Array(user, movie, rating) => Rating(user.toInt, movie.toInt, rating.toDouble) }
    ratings.first()

    /*
    Train the ALS model
    rank=50, (features number) => 고유값 지정, 이것도 svd 처럼 3개 행렬로 분리되네
    943*50, 50*50, 60*1682
    iterations=10,
    lambda=0.01 */
    val model = ALS.train(ratings, 50, 10, 0.01)
    model.userFeatures
    // model: org.apache.spark.mllib.recommendation.MatrixFactorizationModel
    // res29: org.apache.spark.rdd.RDD[(Int, Array[Double])] = FlatMappedRDD[1099] at flatMap at ALS.scala:231


    /* Count user factors and force computation */
    model.userFeatures.count
    // res30: Long = 943
    model.productFeatures.count
    // res31: Long = 1682



    /* Use model to predict */
    /* Make a prediction for a single user (user 789) and movie(movie 123) pair */
    /* predictedRating: Double = 3.12 <-- */

    //789 유저의 평점
    rawRatings.filter(v=> v(0).toInt == 789).foreach(v => println(v(0) +"_"+ v(1) + "_" + v(2)))

    // 평점이 없는 경우
    rawRatings.filter(v=> v(0).toInt == 789).filter(v=> v(1).toInt == 123).foreach(v => println(v(0) +"_"+ v(1) + "_" + v(2)))
    val predictedRating3 = model.predict(789, 123)

    // 기존에 평점이 있는 경우 : 4점
    rawRatings.filter(v=> v(0).toInt == 789).filter(v=> v(1).toInt == 1008).foreach(v => println(v(0) +"_"+ v(1) + "_" + v(2)))
    val predictedRating4 = model.predict(789, 1008)



    /* Make predictions for a single user across all movies */
    val userId = 789
    val K = 10
    val topKRecs = model.recommendProducts(userId, K)
    println(topKRecs.mkString("\n"))
    /*
    Rating(789,693,5.721600308387292)
    Rating(789,182,5.601578586638242)
    Rating(789,135,5.373424844124324)
    Rating(789,156,5.366215654616665)
    Rating(789,192,5.258969602664706)
    Rating(789,603,5.136510434627857)
    Rating(789,474,5.091548710976856)
    Rating(789,650,5.089802631430172)
    Rating(789,23,5.080381811847257)
    Rating(789,179,5.056950756125383)
    */

    // 그런데 평점을 안남기는 사람들이 많으므로... 영화를 식별해서 뭔가 묶어 주자
    /* Load movie titles to inspect the recommendations */
    val movies = sc.textFile("D:\\ml-100k\\u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(array => (array(0).toInt, array(1))).collectAsMap()
    titles(123)
    // res68: String = Frighteners, The (1996)


    val moviesForUser = ratings.keyBy(_.user).lookup(789)
    // moviesForUser: Seq[org.apache.spark.mllib.recommendation.Rating] = WrappedArray(Rating(789,1012,4.0), Rating(789,127,5.0), Rating(789,475,5.0), Rating(789,93,4.0), ...
    // ...
    println(moviesForUser.size)
    // 33 이 사용자는 평점을 33개 남겼구나...
    // take(10) 시점에 드라이버로 데이터 넘어옴
    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product), rating.rating)).foreach(println)
    /* 평점이 높은것 순ㅇ로 10개 표시
    (Godfather, The (1972),5.0)
    (Trainspotting (1996),5.0)
    (Dead Man Walking (1995),5.0)
    (Star Wars (1977),5.0)
    (Swingers (1996),5.0)
    (Leaving Las Vegas (1995),5.0)
    (Bound (1996),5.0)
    (Fargo (1996),5.0)
    (Last Supper, The (1995),5.0)
    (Private Parts (1997),4.0)
    */


    // 위에서 나온 평점 순위들에 영화이름만 붙여줌
    topKRecs.map(rating => (titles(rating.product), rating.rating)).foreach(println)
    /*
    (Casino (1995),5.721600308387292)
    (GoodFellas (1990),5.601578586638242)
    (2001: A Space Odyssey (1968),5.373424844124324)
    (Reservoir Dogs (1992),5.366215654616665)
    (Raging Bull (1980),5.258969602664706)
    (Rear Window (1954),5.136510434627857)
    (Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963),5.091548710976856)
    (Seventh Seal, The (Sjunde inseglet, Det) (1957),5.089802631430172)
    (Taxi Driver (1976),5.080381811847257)
    (Clockwork Orange, A (1971),5.056950756125383)
    */



    // 아이템과 아이템의 유사도
    /* Compute item-to-item similarities between an item and the other items */

    val aMatrix = new DoubleMatrix(Array(1.0, 2.0, 3.0))
    // aMatrix: org.jblas.DoubleMatrix = [1.000000; 2.000000; 3.000000]


    // 코사인 유사도 계산
    /* Compute the cosine similarity between two vectors */
    def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
      vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
    }


    // compute similarity for item id :567
    val itemId = 567
    val itemFactor = model.productFeatures.lookup(itemId).head
    // itemFactor: Array[Double] = Array(0.15179359424040248, -0.2775955241896113, 0.9886005994661484, ...
    val itemVector = new DoubleMatrix(itemFactor)
    // itemVector: org.jblas.DoubleMatrix = [0.151794; -0.277596; 0.988601; -0.464013; 0.188061; 0.090506; ...
    cosineSimilarity(itemVector, itemVector)
    // res113: Double = 1.0000000000000002



    val sims = model.productFeatures.map{ case (id, factor) =>
      val factorVector = new DoubleMatrix(factor)
      val sim = cosineSimilarity(factorVector, itemVector)
      (id, sim)
    }
    val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    // sortedSims: Array[(Int, Double)] = Array((567,1.0), (672,0.483244928887981), (1065,0.43267674923450905), ...
    println(sortedSims.mkString("\n"))
    /*
    (567,1.0000000000000002)
    (1471,0.6932331537649621)
    (670,0.6898690594544726)
    (201,0.6897964975027041)
    (343,0.6891221044611473)
    (563,0.6864214133620066)
    (294,0.6812075443259535)
    (413,0.6754663844488256)
    (184,0.6702643811753909)
    (109,0.6594872765176396)
    */


    /* We can check the movie title of our chosen movie and the most similar movies to it */
    println(titles(itemId))
    // Wes Craven's New Nightmare (1994)
    val sortedSims2 = sims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim) }.mkString("\n")
    /*
    (Hideaway (1995),0.6932331537649621)
    (Body Snatchers (1993),0.6898690594544726)
    (Evil Dead II (1987),0.6897964975027041)
    (Alien: Resurrection (1997),0.6891221044611473)
    (Stephen King's The Langoliers (1995),0.6864214133620066)
    (Liar Liar (1997),0.6812075443259535)
    (Tales from the Crypt Presents: Bordello of Blood (1996),0.6754663844488256)
    (Army of Darkness (1993),0.6702643811753909)
    (Mystery Science Theater 3000: The Movie (1996),0.6594872765176396)
    (Scream (1996),0.6538249646863378)
    */


    // 이제 성능을 측정해보자.
    /* measure of prediction performance */
    /* Compute squared error between a predicted and actual rating */
    // first rating for user 789
    val actualRating = moviesForUser.take(1)(0) // 실제 사용자가 입력한 평점
    // actualRating: Seq[org.apache.spark.mllib.recommendation.Rating] = WrappedArray(Rating(789,1012,4.0))
    // user - item : mean rating is 4


    val predictedRating2 = model.predict(789, actualRating.product)
    // predictedRating: Double = 4.001005374200248 <-- compare with above result
    val squaredError = math.pow(predictedRating2 - actualRating.rating, 2.0) // 잔차의 제곱 계산
    // squaredError: Double = 1.010777282523947E-6



    /* Compute Mean Squared Error (MSE) across the dataset */
    // Apache Spark MLlib guide at: http://spark.apache.org/docs/latest/mllib-guide.html#collaborative-filtering-1
    // 유저와 상품만 빼고
    val usersProducts = ratings.map{ case Rating(user, product, rating)  => (user, product)}
    // 예측치를 key(유저,영화) value 형태로 만든다
    val predictions = model.predict(usersProducts).map{
      case Rating(user, product, rating) => ((user, product), rating)
    }
    // 실제 평점도 마찬가지로 key value로
    val ratingsAndPredictions = ratings.map{
      case Rating(user, product, rating) => ((user, product), rating)
    }.join(predictions)
    val MSE = ratingsAndPredictions.map{
      case ((user, product), (actual, predicted)) =>  math.pow((actual - predicted), 2)
    }.reduce(_ + _) / ratingsAndPredictions.count // reduce(_ + _) => 그냥 다 더하네
    println("Mean Squared Error = " + MSE) // 전데 데이터의 잔차
    // Mean Squared Error = 0.08231947642632856
    val RMSE = math.sqrt(MSE) // 평균적으로 0.3정도 차이남
    println("Root Mean Squared Error = " + RMSE)
    // Root Mean Squared Error = 0.28691370902473196

    // 이건 강화학습과 비슷하다. 일부 데이터로 학습한거다.



    /* Compute Mean Average Precision at K */

    /* Function to compute average precision given a set of actual and predicted ratings */
    // Code for this function is based on: https://github.com/benhamner/Metrics
    def avgPrecisionK(actual: Seq[Int], predicted: Seq[Int], k: Int): Double = {
      val predK = predicted.take(k)
      var score = 0.0
      var numHits = 0.0
      for ((p, i) <- predK.zipWithIndex) {
        if (actual.contains(p)) {
          numHits += 1.0
          score += numHits / (i.toDouble + 1.0)
        }
      }
      if (actual.isEmpty) {
        1.0
      } else {
        score / scala.math.min(actual.size, k).toDouble
      }
    }
    val actualMovies = moviesForUser.map(_.product)
    // actualMovies: Seq[Int] = ArrayBuffer(1012, 127, 475, 93, 1161, 286, 293, 9, 50, 294, 181, 1, 1008, 508, 284, 1017, 137, 111, 742, 248, 249, 1007, 591, 150, 276, 151, 129, 100, 741, 288, 762, 628, 124)


    val predictedMovies = topKRecs.map(_.product)
    // predictedMovies: Array[Int] = Array(27, 497, 633, 827, 602, 849, 401, 584, 1035, 1014)


    val apk10 = avgPrecisionK(actualMovies, predictedMovies, 10)
    // apk10: Double = 0.0


    /* Compute recommendations for all users */
    val itemFactors = model.productFeatures.map { case (id, factor) => factor }.collect()
    val itemMatrix = new DoubleMatrix(itemFactors)
    println(itemMatrix.rows, itemMatrix.columns)
    // (1682,50)


    // broadcast the item factor matrix
    val imBroadcast = sc.broadcast(itemMatrix)



    // compute recommendations for each user, and sort them in order of score so that the actual input
    // for the APK computation will be correct
    val allRecs = model.userFeatures.map{ case (userId, array) =>
      val userVector = new DoubleMatrix(array)
      val scores = imBroadcast.value.mmul(userVector)
      val sortedWithId = scores.data.zipWithIndex.sortBy(-_._1)
      val recommendedIds = sortedWithId.map(_._2 + 1).toSeq
      (userId, recommendedIds)
    }


    // next get all the movie ids per user, grouped by user id
    val userMovies = ratings.map{ case Rating(user, product, rating) => (user, product) }.groupBy(_._1)
    // userMovies: org.apache.spark.rdd.RDD[(Int, Seq[(Int, Int)])] = MapPartitionsRDD[277] at groupBy at <console>:21


    // finally, compute the APK for each user, and average them to find MAPK

    val MAPK = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
      avgPrecisionK(actual, predicted, K)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision at K = " + MAPK)
    // Mean Average Precision at K = 0.030486963254725705



    /* Using MLlib built-in metrics */

    // MSE, RMSE and MAE
    import org.apache.spark.mllib.evaluation.RegressionMetrics
    val predictedAndTrue = ratingsAndPredictions.map { case ((user, product), (actual, predicted)) => (actual, predicted) }
    val regressionMetrics = new RegressionMetrics(predictedAndTrue)
    println("Mean Squared Error = " + regressionMetrics.meanSquaredError)
    println("Root Mean Squared Error = " + regressionMetrics.rootMeanSquaredError)
    // Mean Squared Error = 0.08231947642632852
    // Root Mean Squared Error = 0.2869137090247319



    // MAPK
    import org.apache.spark.mllib.evaluation.RankingMetrics
    val predictedAndTrueForRanking = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2)
      (predicted.toArray, actual.toArray)
    }
    val rankingMetrics = new RankingMetrics(predictedAndTrueForRanking)
    println("Mean Average Precision = " + rankingMetrics.meanAveragePrecision)
    // Mean Average Precision = 0.07171412913757183



    // Compare to our implementation, using K = 2000 to approximate the overall MAP
    val MAPK2000 = allRecs.join(userMovies).map{ case (userId, (predicted, actualWithIds)) =>
      val actual = actualWithIds.map(_._2).toSeq
      avgPrecisionK(actual, predicted, 2000)
    }.reduce(_ + _) / allRecs.count
    println("Mean Average Precision = " + MAPK2000)
    // Mean Average Precision = 0.07171412913757186




  }
}
