package com.insam.Mllib

import org.apache.spark.sql.SparkSession

object mcSpark_1_spark {
  val spark = SparkSession.builder().master("local").getOrCreate()


  import spark.implicits._


  // 무비렌즈 데이터
  val user_data = spark.read.format("csv").option("delimiter","|").load("D:\\ml-100k\\u.user")
  user_data.first()
  user_data.foreach(println(_))
  user_data.show()
  user_data.printSchema()

  user_data.registerTempTable("users")

  spark.sql("select * from users where _c1>40").show()






}
