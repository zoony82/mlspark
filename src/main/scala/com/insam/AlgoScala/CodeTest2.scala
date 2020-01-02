package com.insam.AlgoScala

import scala.collection.mutable.ArrayBuffer

object CodeTest2 {
  def main(args: Array[String]): Unit = {

  }

  def solution(): Unit = {
    val arr = Array(0,9,0,2,6,8,0,8,3,0)

    //잭이 갈수 있는 최대 도시 수를 구하라
    // 키는 갈수 있는 경우의 수를 전부 나열하고, 폴더 레프트로 더하면서 조건 준다? ㅋㅋ

    val path = ArrayBuffer[Any]()
    var ticket_yn = 1

    // 0에서 갈수 있는 길 : 2,6,9
    var hubo_arr_1 = arr.zipWithIndex.filter(v => v._1 == 0 && v._2 != 0).map(v => (0, v._2)).toList


    // 그다음 리스트에서 갈수 있는 길 : 1,3,4
    var hubo_arr_2 = arr.zipWithIndex.filter(v => hubo_arr_1.exists(x => x._2 == v._1)).map(v => (v._1, v._2)).toList

    // 갈수 있는 길에서 처음번째의 길로부터 map 생성
    (hubo_arr_1 ++ hubo_arr_2.map(v=>(v._2,v._1))).groupBy(_._2)
    // 이걸 계속 돌리다보면 hubo_arr 가 null이 나올때까지 돌린다.

    var a = 0;
    var hubo_arr = List((0, 0))
    while( a < 7 ){
      println(a)
      hubo_arr = arr.zipWithIndex.filter(v => hubo_arr.exists(x => x._2 == v._1 && v._2 != 0)).map(v => (v._1, v._2)).toList
      a = a + 1;
      hubo_arr.foreach(v => println(v._1," : ", v._2))
    }



    // 그러면서 결과를 계속 이어간다.
    // 그 결과물을 폴더 left 하자.

  }
}
