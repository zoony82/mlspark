package com.insam.AlgoScala

object PermMissingElem {
  def run(): Unit ={
    val arr = List(2, 3, 5, 6, 1)
    val sort = arr.sortWith((v1,v2) => v1<v2)
    sort.sliding(2).foreach(v =>
      if(v(1)-v(0) != 1) println(v(0)+1))

    // foldLeft 는 누적합을 가지고 연산하기 때문에, 해당 문제에 부적합
    /*
    val result =  sort.foldLeft(0){
      (v1,v2) =>
        println(v1, v2)
        if(v2-v1 != 1) v2-1 else 0
    }

    sort.foldLeft(0){
      (v1,v2) =>
        println(v1, v2)
        v1+v2
    }
     */
  }
}
