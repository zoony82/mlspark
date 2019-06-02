package com.insam.Grammar

object objectTestApply{
  def main(args : Array[String]): Unit ={
    println("Trait Test")
    println(addOne(1))
  }
}

object addOne extends Function1[Int,Int] {

  //Function1 트레잇에는 apply 가 정의되어 있음
  def apply(m:Int) : Int = m + 1
}

