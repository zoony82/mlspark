package com.insam.Fp

import scala.annotation.tailrec

object SktYoutubeSeminar {
  @tailrec
  def getMaxValue(list:List[Int], result:Int): Int ={
    if(list.length == 1){
      return result
    } else if(list(0) > result){
      getMaxValue(list.drop(1),list(0))
    } else{
      getMaxValue(list.drop(1),result)
    }

  }

  def declareFunctionType(a:Int, b:Int): Int ={
    val functionType = (v1:Int, v2:Int) => {v1 * v2}

    val result = functionType(a,b)

    result

  }

  def compareDefValFunction(): Unit ={
    val test:()=>Int ={
      val r = util.Random.nextInt
      ()=>r
    }
    println("val1 : " + test())
    println("val2 : " + test())

    def test2:()=>Int ={
      val r = util.Random.nextInt
      ()=>r
    }
    println("def1 : " + test2())
    println("def2 : " + test2())
  }

  def something():Int={
    println("something declared")
    369
  }
  def callByValue(x:Int)= {
    println("callByValue" + x)
    println("callByValue" + x)
  }
  def callByName(x: => Int)={
    println("callByName" + x)
    println("callByName" + x)
  }
  def callByTest(): Unit ={
    callByValue(something()) // 미리 정의
    callByName(something()) // 호출시마다 정의 => 나중에 복잡한 로직을 필요시에만 정의할때 매우 유용하다.
  }

  def callByTest2()={
    if(false){
      callByValue(something())
      callByName(something())
    }
  }

  def PatialFunctionTest(): Unit ={
    //https://knight76.tistory.com/entry/scala-%EB%B6%80%EB%B6%84-%ED%95%A8%EC%88%98-%EC%BB%A4%EB%A7%81
    val one:PartialFunction[Int,String]={
      case 1 => "one"
    }

    println(one.isDefinedAt(1))
    println(one.isDefinedAt(2))
    println(one(1))
//    println(one(2))
  }


  def main(args: Array[String]): Unit = {
    println("꼬리재귀함수로 최대값 찾기")
    val tempList = List(1,3,4,12,4,56,1,6)
    val maxValue = getMaxValue(tempList, 1)
    println(maxValue)

    println("function type 으로 곱셈")
    println(declareFunctionType(3,6))

    println("val/def 함수의 특징 : val->정의시 평가, def->호출시 평가")
    compareDefValFunction()

    println("callByName 테스트")
    callByTest()

    println("callByName 테스트2")
    callByTest2()

    println("부분함수")
    // 부분함수와 혼동하지 말자 : https://m.blog.naver.com/PostView.nhn?blogId=jjoommnn&logNo=220292065520&proxyReferer=https%3A%2F%2Fwww.google.com%2F
    PatialFunctionTest()

    println("부분적용함수")
    // todo : https://www.youtube.com/watch?v=OuR2mTnxIVA&t=24s




  }
}
