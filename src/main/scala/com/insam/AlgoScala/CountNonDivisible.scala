package com.insam.AlgoScala

object CountNonDivisible {
  def run(): Unit ={

    val test = Array(3,1,2,3,6)
    solution(test)
  }

  def solution(arr : Array[Int]): Array[Int]={
    val arr = Array(3,1,2,3,6)

    //solution 1. general loop
    arr.map(v => {
      arr.filter(x => v%x != 0).length
    })
  }
}
