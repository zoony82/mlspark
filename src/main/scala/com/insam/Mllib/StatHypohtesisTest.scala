package com.insam.Mllib

import org.apache.spark.ml.linalg.{Vectors,Vector}
import org.apache.spark.ml.stat.ChiSquareTest

object StatHypohtesisTest {
  // 가설검증 : 카이제곱검정 - 그룹간 차이가 있는지(=그룹끼리 독립이 아닌지의 여부) 가설검정 하는 방법
  def main(args: Array[String]): Unit = {

    val spark = org.apache.spark.sql.SparkSession.builder().master("local").appName("mllibTest").getOrCreate()


    val data = Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(1.5, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(3.5, 30.0)),
      (0.0, Vectors.dense(3.5, 40.0)),
      (1.0, Vectors.dense(3.5, 40.0))
    )

    println(data)

    import spark.implicits._

    val df = data.toDF("label","features")

    df.show()

    val chi = ChiSquareTest.test(df, "features","label").head()

    println(s"pValues = ${chi.getAs[Vector](0)}")
    println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
    println(s"statistics ${chi.getAs[Vector](2)}")

  }
}
