package com.insam.Mllib

import org.apache.spark.{SparkConf, SparkContext}

object mcSpark_1_Basic {
  def main(args: Array[String]): Unit = {
    /*
  /usr/hdp/2.6.1.0-129/spark2/bin/spark-shell --master yarn-client --num-execu--executor-memory 512M
  => 클러스터 메모리 1gb 먼저 설정 후 각 1g씩 3개를 할당해줘서 총 4gb를 먹더라
   */

    //List(1,2,3) // 한쪽에 몰려있는 데이터가 만들어진 것이다.
    // 분산되게 만드려면...


    //
    //  val list = List(1,2,3)
    //  def addTwo(x:Int) = x+2
    //  list.map(addTwo)



    val conf:SparkConf = new SparkConf().setAppName("Histogram").setMaster("local[4]")
    val sc= new SparkContext(conf)
    //  sc.parallelize(List(1,2,3))

    //  val lines =sc.textFile("D:\\kisang.txt")
    //  lines.foreach(println(_))

    //  val counts = lines.flatMap(v => v.split(" ")).map(v => (v,1)).reduceByKey((a,b)=>(a + b))
    //  counts.foreach(println(_))


    //07.02 무비렌즈 데이터
    val movie_data = sc.textFile("/home/jjh/문서/dataset/ml-100k/u.item")

    def convert_year(x:String) : String = (
      try{
        x.substring(7,11)
      } catch{
        case e:Exception => return "1900"
      }
      )

    val movie_fields = movie_data.map(v => v.split("\\|"))
    movie_fields.first()
    val years = movie_fields.map(v=>v(2)).map(v=>convert_year(v))
    years.count() // 1682
    val year_filtered = years.filter(v => v!="1900")
    year_filtered.count() // 1680
    val movie_ages = year_filtered.map(v => (1988 - v.toInt, 1)).
      reduceByKey(_ + _)
    // 파일이 Executor가 3개임에도 불구하고 데이터가 적어서 2개만 나온다.
    // reduceByKey(_ + _, 5) 로 하면 파일이 5개로 떨어진다.

    val movie_ages_ = movie_ages.map(v => "%d,%d".format(v._1, v._2))
    // 그런데 df()로 변환할대는 괄호를 빼지 않고 변환해야 컬럼으로 변환 된다.

    movie_ages_.foreach(println(_))
    // println 그런데 이거는 제플린에서 쓰면 안나온다
    // 이거를 보려면 드라이버 쪽으로 모아야 한다.
    // movie_ages_.collect().foreach(println(_))


    // 제플린에서 조회하기
    // %sh
    // hdfs dfs -cat /user/zeppelin/movie_ages/part-00000



  }

}
