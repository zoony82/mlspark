package com.insam.Grammar

object objectTest {
  def main(args : Array[String]):Unit = {
    println("object test")
    println(Timer.currentCount()) // new 생성 필요가 없군...
    println(Timer.currentCount())
  }
}

object Timer{
  var count = 0

  def currentCount():Long={
    count +=1
    return count
  }
}