package com.insam.AlgoScala

object CycleRotation {
  def run(): Unit ={
    val arr1 = List(3,4,5,6,7,8,9,10,11,12,13)
    val nValue = 3
    val map1 = arr1.zipWithIndex.toMap
    val resultMap =map1.map(v=>
      (v._1, if(v._2 + nValue >= arr1.length) v._2 + nValue - arr1.length else v._2 + nValue))
    val resultSeq = resultMap.toSeq
    // value기준 오름차순
    val orderSeq = resultSeq.sortWith((t1,t2) => t1._2 < t2._2)
    println(orderSeq.map(v=> v._1).toList)
  }
}
