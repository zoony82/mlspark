package com.insam.Mllib

import org.apache.spark.ml.linalg.{Matrix,Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row


object CorrelationTest {
  // https://spark.apache.org/docs/2.3.0/ml-statistics.html
  // https://gomguard.tistory.com/173
  // https://alphahackerhan.tistory.com/20


  val spark = org.apache.spark.sql.SparkSession.builder().master("local").appName("mllibTest").getOrCreate()

  val data = Seq(
    Vectors.sparse(4, Seq((0, 1.0),(3,-2.0))), // 오직 0이 아닌 값과 그것들의 인덱스를 저장, 벡터의 크기가 4이고 0이 아닌 값들의 위치와 값을 표시
    Vectors.dense(4.0, 5.0, 0.0, 3.0), // 모든 데이터를 부동 소수의 배열에 저장
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
  )

  import spark.implicits._

  val df = data.map(Tuple1.apply).toDF("features")
  df.show()

  val Row(coeff1:Matrix) = Correlation.corr(df, "features").head

  val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head

  println(s"Pearson correlation matrix:\n $coeff1")
  println(s"Spearman correlation matrix:\n $coeff2")


  if(true){
    val data2 = Seq(
      Vectors.dense(1,2,3,4),
      Vectors.dense(5,4,3,2),
      Vectors.dense(5,0,0,0),
      Vectors.dense(12,14,16,20)
    )
    val df2 = data2.map(Tuple1.apply).toDF("feat")
    df2.show()
    val result = Correlation.corr(df2,"feat").head
    println(result)




    val data3 = Seq(
      Vectors.dense(1,20,0,-10),
      Vectors.dense(5,40,0,-20),
      Vectors.dense(7,76,0,-40),
      Vectors.dense(12,90,0,-70),
      Vectors.dense(12,1000,0,-90)
    )
    val df3 = data3.map(Tuple1.apply).toDF("feat")
    df2.show()
    val result1 = Correlation.corr(df3,"feat").head
  }

}
