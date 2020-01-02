package com.insam.AlgoScala

object TennisTournament {
  def run(): Unit ={
    val a = soultion(5,3)
    val b = soultion(10,3)
    val c = soultion(3,5)

  }

  def soultion(p:Int, c:Int):Int = {
    if(p > c*2){
      c
    } else{
      p/2
    }
  }
}
