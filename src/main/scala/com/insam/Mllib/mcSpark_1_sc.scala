package com.insam.Mllib

import org.apache.spark.{SparkConf, SparkContext}

object mcSpark_1_sc {

  /*
  /usr/hdp/2.6.1.0-129/spark2/bin/spark-shell --master yarn-client --num-execu--executor-memory 512M
  => 클러스터 메모리 1gb 먼저 설정 후 각 1g씩 3개를 할당해줘서 총 4gb를 먹더라
   */

  List(1,2,3) // 한쪽에 몰려있는 데이터가 만들어진 것이다.
  // 분산되게 만드려면...



  val list = List(1,2,3)
  def addTwo(x:Int) = x+2
  list.map(addTwo)



  val conf:SparkConf = new SparkConf().setAppName("Histogram").setMaster("local")
  val sc= new SparkContext(conf)
  sc.parallelize(List(1,2,3))

  val lines =sc.textFile("D:\\kisang.txt")
  lines.foreach(println(_))

  val counts = lines.flatMap(v => v.split(" ")).map(v => (v,1)).reduceByKey((a,b)=>(a + b))
  counts.foreach(println(_))



}
