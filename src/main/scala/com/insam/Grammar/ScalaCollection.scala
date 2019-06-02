package com.insam.Grammar

import org.apache.spark

object ScalaCollection {
  def main(args: Array[String]){

    //http://twitter.github.io/scala_school/ko/collections.html
    println("Scala School Collection")
    println("=================\n")

    println("Basic Data Structure")
    println("=================\n")

    println("\nList\n=========")
    val vlist = List(1,2,3,4)
    println(vlist)
    val vlist2 = List(1,4,3,2)
    println(vlist2)

    println("\nSet\n=========")
    val vset1 = Set(1,2)
    println(vset1)
    val vset2 = Set(1,2,2)
    println(vset2)

    println("\nTuple\n=========")
    val vtuple = ("localhost",80)
    println(vtuple)
    println(vtuple._1, vtuple._2)

    println("\nMap\n=========")
    val vmap1 = Map(1->2, 1->4)
    println(vmap1)
    val vmap2 = Map(1->2, 3->4)
    println(vmap2)
    val vmap3 = Map(1->2, 3->"abc")
    println(vmap3)

    println("Function Combination")
    println("=================\n")

    println("\nmap\n=========")
    val vclist = List(1,2,3,4)
    val vclist2 = vclist.map((i:Int)=> i*2)
    println(vclist2)

    println("\nforeach\n=========")
    val flist = List(1,2,3,4)
    val flist2 = flist.foreach((i:Int)=> i*2)
    println(flist2) // foreach 는 반환을 안함
    println(flist) // 뭐야 저장도 안하네... 그냥 돌리기만 하나보네

    println("\nfilter\n=========")
    val filterlist = List(1,2,3,4)
    val filterlistResult = filterlist.filter((i:Int) => i%2 == 0)
    println(filterlistResult)

    println("\nzip\n=========")
    println(List(1,2,3).zip(List("a","b","c")))
    println(List(1,2,3).zip(List("a","b","c")).zip(List("d","e","f")))

    println("\npartition\n=========")
    println(List(1,2,3,4,5,6,7,8,9,10).partition(_%2 ==0))

    println("\nfind\n=========")
    val findlist = List(1,2,3,4)
    val findlistResult = filterlist.find((i:Int) => i>2)
    println(findlistResult)

    println("\ndrop/dropWhile\n=========")
    println(List(1,2,3,4,5).drop(3))
    println(List(1,2,3,4,5).dropWhile(_%2!=0))

    println("\nfoldLeft\n=========")
    println(List(1,2,3,4,5).foldLeft(0)((m:Int, n:Int)=> m+n))

    println("\nfoldRight\n=========")
    println(List(1,2,3,4,5).foldRight(0)((m:Int, n:Int)=> m+n))

    println("\nflatten\n=========")
    println(List(List(1,2),List(3,4)).flatten)

    println("\nflatmap : map+flatten\n=========")
    val nested = List(List(1,2),List(3,4))
    val nested2 = nested.flatMap(x => x.map(_*2))
    println(nested2)

    println("\nMap?\n=========")
    // All of the function combinators shown above are also available in the map.
    val ext = Map("steve"->100,"bob"->101,"joe"->201)
    println(ext.filter((x:(String,Int)) => x._2 < 200))
    println(ext.filter((x:(String,Int)) => x._1 == "bob"))
    //You can easily separate keys and values using pattern matching.
    println(ext.filter({case(name,extension) => extension<200}))
    println(ext.filter({case(name,extension) => name=="bob"}))



  }
}
