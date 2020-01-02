package com.insam.AlgoScala

object SocksLaundering {
  def run(): Unit ={
    println(solution(3,Array(1,2,1,4),Array(1,2,5,4)))

  }

  def main(args: Array[String]): Unit = {
    println(solution(2,Array(1, 2, 1, 1),Array(1, 4, 3, 2, 4)))
    println(solution(3,Array(1,2,1,4),Array(1,2,5,4)))
    println(solution(10,Array(1,2,1,4,2,1,3,3,2,5,4,3,2,1,3,4,3),Array(1,2,5,4,1,2,3,3,2,2,3,4,2,2,3,4,2)))

  }

  def solution(k:Int, c:Array[Int], d:Array[Int]): Int ={

    val k_cnt = k //10
    val c_arr = c //Array(1,2,1,4,2,1,3,3,2,5,4,3,2,1,3,4,3)
    val d_arr = d // Array(1,2,5,4,1,2,3,3,2,2,3,4,2,2,3,4,2)
    /*
    val k_cnt = 2
    val c_arr = Array(1, 2, 1, 1)
    val d_arr = Array(1, 4, 3, 2, 4)
     */
    var reamin_k_cnt = k
    var max_socks = 0

    //step1. 깨끗한 양말 중에 몇개 신을 수 있는가. -> 카운팅
    val c_arr_group = c_arr.groupBy(identity).mapValues(v => v.length)
    // or c_arr.groupBy(identity).map(v => (v._1,v._2.length))
    val c_arr_group_select = c_arr_group.filter(v => v._2 /2 >= 1 ).map(v => (v._1, v._2 /2))
    max_socks = c_arr_group_select.foldLeft(0)((a,b) => a+b._2) //★ 양말개수 획득
    val c_arr_remain = c_arr_group.filter(v => v._2 %2 == 1).map(v => (v._1, { if(v._2 %2 == 1){1}else{0} }) )

    //step2. 홀수 인 양말과 1개씩 추가로 존재하는 더티한 양말은 있는가. -> 카운팅 (세탁기 초과하면 종료)
    val d_arr_group = d_arr.groupBy(identity).mapValues(v => v.length)

    // 전체 합치기
    val cd_arr_sum_all = c_arr_remain ++ d_arr_group.map { case (k,v) => k -> (v + c_arr_remain.getOrElse(k,0)) }

    // 깨끗한 양말하고만 합치기
    // 깨끗한 양말이 있는지 개수 세기
    val c_arr_remain_list = c_arr_remain.map(v => v._1)
    val result_c_d = d_arr_group.filter(v => c_arr_remain_list.exists(x => x==v._1)).map(v=>(v,1)).foldLeft(0)((a,b) => (a + b._2)) //★ 양말개수 획득

    // 세탁개 개수 확인하기
    if(result_c_d < k_cnt){
      max_socks = max_socks + result_c_d
      reamin_k_cnt = k_cnt - result_c_d
    } else{
      max_socks = max_socks + k_cnt
      reamin_k_cnt = 0
    }

    //step3. 더티한 양말 중 짝수개로 존재하는 양말이 있는가. -> 카운팅 (세탁기 초과하면 종료)
    // reamin_k_cnt 가 존재 할때 진행
    if(reamin_k_cnt>0){
      // 깨끗한 세탁물과 일치하는거 1개씩 빼기
      val d_arr_group_reamin = d_arr_group.map(v => (v._1, if(c_arr_remain_list.exists(x => x==v._1)){v._2-1}else{v._2}))
      // 더러운 양말 중 세탁 가능한 대상
      val result_d = d_arr_group_reamin.map(v =>(v._1, v._2/2)).foldLeft(0)((a,b) => a + b._2)

      // 세탁개 개수 확인하기
      if(result_d < reamin_k_cnt){
        max_socks = max_socks + result_d
        reamin_k_cnt = reamin_k_cnt - result_d
      } else{
        max_socks = max_socks + reamin_k_cnt
        reamin_k_cnt = 0
      }

    }

    max_socks

  }
}
