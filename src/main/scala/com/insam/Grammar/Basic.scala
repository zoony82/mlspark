package com.insam.Grammar

object Basic {
  def main(args : Array[String]): Unit = {
    println("basic test")
    println(1+1)

    val two = 2
    var name="steve"
    name="joe"

    println(addOne(4))
    println(three)

    //이름없는 함수
    val addTwo = (x:Int) => x + 2
    println(addTwo)
    println(addTwo(2))

    //인자의 일부만 사용해 호출하기(부분 적용, partial application)
    val add2 = adder(20,_:Int)
    println(add2(30))

    //함수의 인자중 일부를 적용하고, 나머지는 나중에 적용하게 남겨두는 경우
    println(multiply(2)(3))
    val timesTwo = multiply(2)_
    println(timesTwo(3))

    //가변 길이 인자 : 동일한 타입의 매개변수가 반복되는 경우를 처리할 수 있는 문법
    val abc = capiAll("rac", "apple")
    println(abc)
  }

  def addOne(m:Int): Int = m + 1
  def three() = 1 + 2
  def adder(m:Int, n:Int) = m + n
  def multiply(m:Int)(n:Int):Int = m * n

  def capiAll(args:String*) = {
    args.map {
      arg => arg.capitalize
    }
  }

}
