package com.insam.AlgoScala

object CodeTest1 {
  def main(args: Array[String]): Unit = {

  }

  def solution(): Unit = {
//    val arr = Array(3,4,5,3,7)
//    val arr = Array(1,2,3,4)
    val arr = Array(1,3,1,2)

    var result=9999

    //한번만 컷팅할 수 있다.
    //슬라이딩 해서 다음 값과의 차를 만든다.

    // 모두 이쁜지 검사
    if(arr.sliding(3).filter(v => ((v(0)-v(1)) * (v(1)-v(2)) > 0) ).toList.length == 0){
      result = 0
    }

    // 이쁘지 않으면 개수 세기
    if(result != 0){
      //step1. 하나씩 빼면서 n(배열 개수)개의 열을 만든다.
      val arr_full = (1 to arr.length).map(v => (arr.slice(0,arr.length-v) ++ arr.slice(arr.length-v+1,arr.length))).toArray


      //step2. 슬라이딩의 차를 계산하고, 총 합이 0이 아닌것을 찾는다.
      val arr_full_possible = arr_full.filter(v => {
        v.sliding(3).filter(v => ((v(0)-v(1)) * (v(1)-v(2)) > 0) ).toList.length == 0
      }).length

      // 없으면 -1
      if(arr_full_possible == 0){
        result = -1
      } else{
        result = arr_full_possible
      }
    }

    result


    //    arr_full.foreach(v => v.foreach(x => println(x)))

    //step2. 슬라이딩 으로 만든다.
    /*
    arr.sliding(3).foreach(v => println(v(0),v(1),v(2)))
    val arr_sliding = arr.sliding(3).map(v => ((v(0),v(1),v(2)))).toList
    arr.sliding(3).map(v => ((v(0)-v(1)) * (v(1)-v(2)))).toList
    arr.sliding(3).filter(v => ((v(0)-v(1)) * (v(1)-v(2)) > 0) ).toList.length
    */

  }
}
