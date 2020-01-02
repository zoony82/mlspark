package com.insam.AlgoScala

object test {
  def main(args: Array[String]): Unit = {
//    testFunc()
    val arr = Array(1).distinct
    var not_occur = 0
    val arr_pos = arr.filter(v => v > 0)
    var arr_pos_sort = arr_pos.sortWith((v1,v2) => v1<v2)
    arr_pos_sort.sliding(2).foreach(v => {
      if(v(1) - v(0) != 1) not_occur = (v(1)-1)
    })
    if(arr_pos.length == 0){
      1
    }
    else if(not_occur == 0) {
      arr_pos.reduceLeft(_ max _)+1
    }
    else{
      not_occur
    }

  }

  def testFunc():Unit = {
    println("test")
  }
}
