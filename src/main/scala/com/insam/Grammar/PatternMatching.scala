package com.insam.Grammar

object PatternMatching {
  def main(args : Array[String]) : Unit = {
    val times = 2

    // Value Matching
    times match{
      case 1 => println("one")
      case 2 => println("two")
      case _ => println("some other number")
    }

    // Condition Matching
    times match{
      case i if i == 1 => println("cond one")
      case i if i == 2 => println("cond two")
      case _ => println("cond some other number")
    }

    // Type Matching
    println(bigger(-5))
    println(bigger(5))
    println(bigger(-5.0))
    println(bigger(5.0))
  }

  // 타입이 다른 값을 서로 다른 방식으로 처리
  def bigger(o: Any): Any = {
    o match{
      case i: Int if i < 0 => i - 1
      case i: Int => i + 1
      case d: Double if d < 0.0 => d - 0.1
      case d: Double => d + 0.1
    }
  }

  // 클래스 멤버에 대해 매치도 가능
//  def calcType(calc: Calculator) = calc match {
//    case calc.brand == "HP" && calc.model == "20B" => "financial"
//    case calc.brand == "HP" && calc.model == "48G" => "scientific"
//    case calc.brand == "HP" && calc.model == "30B" => "business"
//    case _ => "unknown"
//  }

}
