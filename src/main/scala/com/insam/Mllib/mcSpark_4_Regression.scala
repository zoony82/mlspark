package com.insam.Mllib

import breeze.linalg.sum
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}


object mcSpark_4_Regression {
  val conf:SparkConf = new SparkConf().setAppName("Histogram").setMaster("local")
  val sc= new SparkContext(conf)

  // header 정보
  // instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt

  val path = "D:\\regression\\hour_noheader.csv"
  val raw_data = sc.textFile(path)
  val num_data = raw_data.count() //17379
  val records = raw_data.map(x => x.split(","))
  records.first()
  println(num_data)


  def get_mapping(rdd:RDD[Array[String]], idx:Int)={
    rdd.map(filed=>filed(idx)).distinct().zipWithIndex().collectAsMap()
  }


  val mappings=for(i<-Range(2,10))yield get_mapping(records,i)
  val cat_len=sum(mappings.map(_.size)) //57개의 카테고리들
  val num_len=records.first().slice(10,14).size //4
  val total_len=cat_len+num_len //61



  val data=records.map{record=>
    val cat_vec=Array.ofDim[Double](cat_len)
    var i=0
    var step=0

    for(filed<-record.slice(2,10)){
      val m=mappings(i)
      val idx=m(filed)
      cat_vec(idx.toInt+step)=1.0
      i=i+1
      step=step+m.size
    }

    val num_vec=record.slice(10,14).map(x=>x.toDouble)
    val features=cat_vec++num_vec
    val label=record(record.size-1).toInt
    LabeledPoint(label,Vectors.dense(features))
  }


  val linear_model=LinearRegressionWithSGD.train(data,10,0.1)
  val true_vs_predicted=data.map(p=>(p.label,linear_model.predict(p.features)))
  println( true_vs_predicted.take(5).toVector.toString())



  val categoricalFeaturesInfo = Map[Int, Int]()
  val linear_model2=DecisionTree.trainRegressor(data,categoricalFeaturesInfo,"variance",5,32)
  val true_vs_predicted2=data.map(p=>(p.label,linear_model.predict(p.features)))
  println( true_vs_predicted2.take(5).toVector.toString())



}
