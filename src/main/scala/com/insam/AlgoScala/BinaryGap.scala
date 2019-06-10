package com.insam.AlgoScala

object BinaryGap {

  def run()={
    println("BinaryGap")
    val input = 124421342
    val binaryStr = input.toBinaryString
    println("input,binaryStr : " + input + " / " + binaryStr)
    var maxValue=0
    (0 to binaryStr.length()-1).foreach{v =>
      if(maxValue < nextOneStr(binaryStr, v)) maxValue = nextOneStr(binaryStr, v)
    }
    println("maxValue : " + maxValue)
  }

  def nextOneStr(binaryStr:String, index:Int): Int ={
    binaryStr.drop(index).indexOfSlice("1")
  }
}
