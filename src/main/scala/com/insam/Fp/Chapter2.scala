package com.insam.Fp

object Chapter2 {
  //2.1 n번째 피보나치수를 돌려주는 재귀함수 : 지역 꼬리 재귀 함수 사용해야 할 것
  println("20190402")

  def fibonacci(n:Int): Int ={
    @annotation.tailrec
    def loop(n:Int, prev:Int, cur:Int): Int ={
      if(n == 0) cur
      else loop(n-1,cur, prev + cur)
    }
    loop(n,0,1)
  }



  println("20190403")

  // 배열에서 한 요소를 찾는 단형성 함수
  def findFirst(ss:Array[String], key:String) : Int ={
    @annotation.tailrec
    def loop(n:Int): Int ={
      if(n > ss.length) -1
      else if(ss(n) == key) n
      else loop(n+1)
    }
    loop(0)
  }

  // 배열에서 한 요소를 찾는 다형성 함수
  def findFirst2[A](as:Array[A], p : A => Boolean) : Int ={
    @annotation.tailrec
    def loop(n:Int): Int ={
      if(n > as.length) -1
      else if(p(as(n))) n
      else loop(n+1)
    }
    loop(0)
  }

  //2.2 array[a]가 주어진 비교 함수에 의거해서 정렬되어 있는지 점검하는 isSorted 함수를 구현하라
  // 서명은 아래와 같다 . def isSorted[A](as:Array[A], ordered:(A,A) => Boolean) : Boolean

  def isSorted[A](as:Array[A], ordered:(A,A) => Boolean) : Boolean ={
    @annotation.tailrec
    def loop(n:Int): Boolean ={
      if(as.length-1 <= n) true
      else if(ordered(as(n),as(n+1))) false
      else loop(n+1)
    }
    loop(0)
  }

  println("20190404")
  //고차 함수 : 부분적용
  def partial1[A,B,C](a:A, f:(A,B) => C) : B => C =
    (b:B) => f(a,b)

  //고차 함수 : 커링
  def curry[A,B,C](f:(A,B) => C) : A => (B => C) =
    a => b => f(a,b)

  //고차 함수 : 언커링
  def uncurry[A,B,C](f:A => B => C) : (A,B) => C =
    (a,b) => f(a)(b)

  //고차 함수 : 합성
  def compose[A,B,C](f:B =>C, g:A =>B) : A => C =
    a=> f(g(a))



  def main(args:Array[String]): Unit ={
//    val a = fibonacci(5)
//    println(a)

    val res = findFirst(Array("a","b","c","d"),"d")
    println(res)

    val res1 = findFirst2(Array("a","b","c","d"),(v : String) => v =="d")
    println(res1)

    val res2 = findFirst2(Array(1,2,3,4),(v : Int) => v ==2)
    println(res2)

    val res3 = isSorted(Array(1,2,3,4),(v1 : Int,v2 : Int) => v1>v2)
    println(res3)

    val res4 = isSorted(Array(1,2,3,4),(v1 : Int,v2 : Int) => v1<v2)
    println(res4)
  }



}
