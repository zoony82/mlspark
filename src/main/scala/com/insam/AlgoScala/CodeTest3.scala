package com.insam.AlgoScala

object CodeTest3 {
  def main(args: Array[String]): Unit = {


  }

  def solution(): Unit = {
    // 분수가 같은것들의 빈도수를 세어라

    val x = Array(1,2,3,4,0)
    val y = Array(2,3,6,8,4)

    val xy = x zip y
    xy.map(v => v._1.toDouble / v._2.toDouble).groupBy(identity).mapValues(_.size).map(v => v._2).max



    
  }
}
