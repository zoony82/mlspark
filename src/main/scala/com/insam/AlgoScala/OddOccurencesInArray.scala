package com.insam.AlgoScala

object OddOccurencesInArray {
  def run(): Unit ={
    println("OddOccurencesInArray")
    val input = Array(9, 3, 9, 3, 9, 7, 9, 8)
    print("input : ")
    input.foreach(v => print(v))
    println()

    val result = input.
      groupBy(v=>v.toString).
      mapValues(v=>v.length).
      filter(v=>v._2 % 2 != 0 ).
      map(v => v._1)

    print("result : ")
    result.foreach(print(_))
  }

}
