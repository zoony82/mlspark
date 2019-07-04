package com.insam.Mllib

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object mcSpark_6_Clustering {
  def main(args: Array[String]): Unit = {
    // https://www.google.com/search?q=kddcup&oq=kddcup&aqs=chrome..69i57j0l5.1573j0j4&sourceid=chrome&ie=UTF-8
    // 센서 데이터에서 비정상 데이터를 어떻게 찾아낼꺼냐
    // kddcup : back,buffer_overflow,ftp_write,guess_passwd,imap,ipsweep,land,loadmodule,multihop,neptune,nmap,normal,perl,phf,pod,portsweep,rootkit,satan,smurf,spy,teardrop,warezclient,warezmaster.
    //

    val conf:SparkConf = new SparkConf().setAppName("Histogram").setMaster("local")
    val sc= new SparkContext(conf)

    val rawData = sc.textFile("D:\\clustering\\kddcup\\kddcup.data")

    // # 1st Clustering #######################################################

    // 각 신호를 명시하는것에 대해 count를 세서 보자
    rawData.map(_.split(',').last).countByValue().toSeq.sortBy(_._2).reverse.foreach(println)
    /*
    (smurf.,280790)
    (neptune.,107201)
    (normal.,97278)
    (back.,2203)
    (satan.,1589)
    (ipsweep.,1247)
    (portsweep.,1040)
    (warezclient.,1020)
    (teardrop.,979)
    (pod.,264)
    (nmap.,231)
    (guess_passwd.,53)
    (buffer_overflow.,30)
    (land.,21)
    (warezmaster.,20)
    (imap.,12)
    (rootkit.,10)
    (loadmodule.,9)
    (ftp_write.,8)
    (multihop.,7)
    (phf.,4)
    (perl.,3)
    (spy.,2)
    # 23 distinct label
    */

    val labelsAndData = rawData.map { line =>
      val buffer = line.split(',').toBuffer
      buffer.remove(1, 3)// 문자 제거
      val label = buffer.remove(buffer.length - 1)
      val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
      (label, vector)
    }

    val data = labelsAndData.values.cache()

    val kmeans = new KMeans()
    val model = kmeans.run(data)

    model.clusterCenters.foreach(println)
    /* two vector printed, it means k=2 clusters */

    // 2개로 묶인게 잘 된걸까
    val clusterLabelCount = labelsAndData.map { case (label, datum) =>
      val cluster = model.predict(datum)
      (cluster, label)
    }.countByValue()

    clusterLabelCount.toSeq.sorted.foreach { case ((cluster, label), count) =>
      println(f"$cluster%1s$label%18s$count%8s")
    }

    data.unpersist()
    // portsweep만 1로 묶였다.
    // 인자를 없이 묶었더니 2개로 묶였는데... 몇개로 묶여야 할까
    /*
    0             back.    2203
    0  buffer_overflow.      30
    0        ftp_write.       8
    0     guess_passwd.      53
    0             imap.      12
    0          ipsweep.    1247
    0             land.      21
    0       loadmodule.       9
    0         multihop.       7
    0          neptune.  107201
    0             nmap.     231
    0           normal.   97278
    0             perl.       3
    0              phf.       4
    0              pod.     264
    0        portsweep.    1039
    0          rootkit.      10
    0            satan.    1589
    0            smurf.  280790
    0              spy.       2
    0         teardrop.     979
    0      warezclient.    1020
    0      warezmaster.      20
    1        portsweep.       1
    res47: data.type = MapPartitionsRDD[236] at values at <console>:61
     */

    // 클러스터의 개수를 바꿔가면서 돌려보면,
    // 섬오브 스퀘어 기울기가 낮아지기 시작하는 부분의 k 개수로 설정을 해야 한다.




    // 적당한 k값 찾기
    //# 2nd clustering #############################################################
    def distance(a: Vector, b: Vector) =
      math.sqrt(a.toArray.zip(b.toArray).map(p => p._1 - p._2).map(d => d * d).sum)

    def distToCentroid(datum: Vector, model: KMeansModel) = {
      val cluster = model.predict(datum)
      val centroid = model.clusterCenters(cluster)
      distance(centroid, datum)
    }

    def clusteringScore(data: RDD[Vector], k: Int): Double = {
      val kmeans = new KMeans()
      kmeans.setK(k) // k값을 바꿔가면서 계산
      val model = kmeans.run(data)
      data.map(datum => distToCentroid(datum, model)).mean()
    }

    def clusteringScore2(data: RDD[Vector], k: Int): Double = {
      val kmeans = new KMeans()
      kmeans.setK(k)
      kmeans.setRuns(1) // 사실 여러번 돌려야 함
      kmeans.setEpsilon(1.0e-6)
      val model = kmeans.run(data)
      data.map(datum => distToCentroid(datum, model)).mean()
    }


    //   # Runs : how many times
    //   # Epsilon : minimum movement centeroid

    val data = rawData.map { line =>
      val buffer = line.split(',').toBuffer
      buffer.remove(1, 3)
      buffer.remove(buffer.length - 1)
      Vectors.dense(buffer.map(_.toDouble).toArray)
    }.cache()

    // 5씩 증가시키면서 테스트
    (5 to 30 by 5).map(k => (k, clusteringScore(data, k))).foreach(println)
    /* 섬오브스퀘어 값이... 15에서 더이상 진전이 없는듯
    (5,1883.6167369867283)
    (10,1779.2255842349878)
    (15,1634.3916047022044)
    (20,1618.092864547269)
    (25,1557.8086040260334)
    (30,1548.760591524399)
     */

    (30 to 100 by 10).par.map(k => (k, clusteringScore2(data, k))).toList.foreach(println)

    /*
    (30,1529.5099752810656)
    (40,1531.0610157137965)
    (50,1160.7294858976188)
    (60,986.0323908108815)
    (70,1464.2209982528486)
    (80,963.9258032200726)
    (90,1530.285185232871)
    (100,910.8536048998116)
     */
    data.unpersist()


    //# R Visualization #########################################################
    val data = rawData.map { line =>
      val buffer = line.split(',').toBuffer
      buffer.remove(1, 3)
      buffer.remove(buffer.length - 1)
      Vectors.dense(buffer.map(_.toDouble).toArray)
    }.cache()

    val kmeans = new KMeans()
    kmeans.setK(100)
    kmeans.setRuns(1)
    kmeans.setEpsilon(1.0e-6)
    val model = kmeans.run(data)

    val sample = data.map(datum =>
      model.predict(datum) + "," + datum.toArray.mkString(",")
    ).sample(false, 0.05)

    sample.saveAsTextFile("hdfs://hatest/user/hadoop/rvis")

    data.unpersist()


    // 비정상 데이터를 찾자
    //    # Anomaly Detect ########################################################
    // z-scored 표준화(standard sacle 이거랑 비슷한거 짠거다)
    def buildNormalizationFunction(data: RDD[Vector]): (Vector => Vector) = {
      val dataAsArray = data.map(_.toArray)
      val numCols = dataAsArray.first().length
      val n = dataAsArray.count()
      val sums = dataAsArray.reduce(
        (a, b) => a.zip(b).map(t => t._1 + t._2))
      val sumSquares = dataAsArray.aggregate(
        new Array[Double](numCols)
      )(
        (a, b) => a.zip(b).map(t => t._1 + t._2 * t._2),
        (a, b) => a.zip(b).map(t => t._1 + t._2)
      )
      val stdevs = sumSquares.zip(sums).map {
        case (sumSq, sum) => math.sqrt(n * sumSq - sum * sum) / n
      }
      val means = sums.map(_ / n)

      (datum: Vector) => {
        val normalizedArray = (datum.toArray, means, stdevs).zipped.map(
          (value, mean, stdev) =>
            if (stdev <= 0)  (value - mean) else  (value - mean) / stdev
        )
        Vectors.dense(normalizedArray)
      }
    }

    // 카테고리컬 한걸 변수화
    def buildCategoricalAndLabelFunction(rawData: RDD[String]): (String => (String,Vector)) = {
      val splitData = rawData.map(_.split(','))
      val protocols = splitData.map(_(1)).distinct().collect().zipWithIndex.toMap
      val services = splitData.map(_(2)).distinct().collect().zipWithIndex.toMap
      val tcpStates = splitData.map(_(3)).distinct().collect().zipWithIndex.toMap
      (line: String) => {
        val buffer = line.split(',').toBuffer

        // 범주형들을 가변수로 변경
        val protocol = buffer.remove(1)
        val service = buffer.remove(1)
        val tcpState = buffer.remove(1)
        val label = buffer.remove(buffer.length - 1)
        val vector = buffer.map(_.toDouble)

        // 원 핫 인코딩
        val newProtocolFeatures = new Array[Double](protocols.size)
        newProtocolFeatures(protocols(protocol)) = 1.0
        val newServiceFeatures = new Array[Double](services.size)
        newServiceFeatures(services(service)) = 1.0
        val newTcpStateFeatures = new Array[Double](tcpStates.size)
        newTcpStateFeatures(tcpStates(tcpState)) = 1.0

        vector.insertAll(1, newTcpStateFeatures)
        vector.insertAll(1, newServiceFeatures)
        vector.insertAll(1, newProtocolFeatures)

        (label, Vectors.dense(vector.toArray))
      }
    }


    // 디시젼트리에서 썻던거
    def entropy(counts: Iterable[Int]) = {
      val values = counts.filter(_ > 0)
      val n: Double = values.sum
      values.map { v =>
        val p = v / n
        -p * math.log(p)
      }.sum
    }

    //
    def buildAnomalyDetector(
                              data: RDD[Vector],
                              normalizeFunction: (Vector => Vector)): (Vector => Boolean) = {
      val normalizedData = data.map(normalizeFunction) // 표준화 시키고
      normalizedData.cache()

      val kmeans = new KMeans()
      kmeans.setK(150)
      kmeans.setRuns(10)
      kmeans.setEpsilon(1.0e-6)
      val model = kmeans.run(normalizedData)

      normalizedData.unpersist()

              // 정규화 해서
      val distances = normalizedData.map(datum => distToCentroid(datum, model))
            // 중심점으로 부터 값이 먼것을 찾자
      val threshold = distances.top(100).last

          // 스레셜드보다 높은것만 뽑는다
      (datum: Vector) => distToCentroid(normalizeFunction(datum), model) > threshold
    }

    def anomalies(rawData: RDD[String]) = {
      val parseFunction = buildCategoricalAndLabelFunction(rawData)
      val originalAndData = rawData.map(line => (line, parseFunction(line)._2))
      val data = originalAndData.values
      val normalizeFunction = buildNormalizationFunction(data)
      val anomalyDetector = buildAnomalyDetector(data, normalizeFunction)
      val anomalies = originalAndData.filter {
        case (original, datum) => anomalyDetector(datum)
      }.keys

      // 비정상 10개만 찍자
      anomalies.take(10).foreach(println)
    }

    anomalies(rawData)
    /*
    normal 같은 정상이 나왔지만 정상이 아니라는 것이다.
    loadmodule, 머 등등이 이런것들 이다.

    1,tcp,smtp,SF,1505,329,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,0,1,2,0.00,0.00,0.00,0.00,1.00,0.00,1.00,16,16,1.00,0.00,0.06,0.00,0.00,0.00,0.00,0.00,normal.
    79,tcp,telnet,SF,281,1301,0,0,0,2,0,1,1,1,0,0,4,2,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,1,10,1.00,0.00,1.00,0.30,0.00,0.00,0.00,0.10,loadmodule.
    25,tcp,telnet,SF,269,2333,0,0,0,0,0,1,0,1,0,2,2,1,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,69,2,0.03,0.06,0.01,0.00,0.00,0.00,0.00,0.00,perl.
    1,tcp,ftp,SF,60,189,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,151,47,0.31,0.03,0.01,0.00,0.00,0.00,0.00,0.00,normal.
    23,tcp,telnet,SF,104,276,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,1,2,1.00,0.00,1.00,1.00,0.00,0.00,0.00,0.00,guess_passwd.
    0,tcp,telnet,S1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1.00,1.00,0.00,0.00,1.00,0.00,0.00,126,1,0.01,0.05,0.01,0.00,0.01,1.00,0.00,0.00,normal.
    0,tcp,pop_3,SF,64,15548,0,0,0,0,0,1,1,1,0,3,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,7,1,0.14,0.29,0.14,0.00,0.00,0.00,0.00,0.00,normal.
    60,tcp,telnet,S3,125,179,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1.00,1.00,0.00,0.00,1.00,0.00,0.00,1,1,1.00,0.00,1.00,0.00,1.00,1.00,0.00,0.00,guess_passwd.
    60,tcp,telnet,S3,126,179,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,2,2,0.50,0.50,0.50,0.50,1.00,0.00,0.00,23,23,1.00,0.00,0.04,0.00,0.09,0.09,0.91,0.91,guess_passwd.
    1,tcp,telnet,SF,123,2053,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,52,52,1.00,0.00,0.02,0.00,0.04,0.04,0.94,0.94,guess_passwd.
     */


  }
}
