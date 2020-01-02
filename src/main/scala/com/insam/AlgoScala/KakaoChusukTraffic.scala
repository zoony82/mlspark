package com.insam.AlgoScala

import java.text.SimpleDateFormat
import java.util.Calendar

object KakaoChusukTraffic {
  def run(): Unit ={

    val test = Array(3,1,2,3,6)
    solution(test)
    /*
    이번 추석에도 시스템 장애가 없는 명절을 보내고 싶은 어피치는 서버를 증설해야 할지 고민이다.
    장애 대비용 서버 증설 여부를 결정하기 위해 작년 추석 기간인 9월 15일 로그 데이터를 분석한 후
    초당 최대 처리량을 계산해보기로 했다.
    초당 최대 처리량은 요청의 응답 완료 여부에 관계없이 임의 시간부터 1초(=1,000밀리초)간
    처리하는 요청의 최대 개수를 의미한다.

    solution 함수에 전달되는 lines 배열은 N(1 ≦ N ≦ 2,000)개의 로그 문자열로 되어 있으며, 각 로그 문자열마다 요청에 대한 응답완료시간 S와 처리시간 T가 공백으로 구분되어 있다.
    응답완료시간 S는 작년 추석인 2016년 9월 15일만 포함하여 고정 길이 2016-09-15 hh:mm:ss.sss 형식으로 되어 있다.
    처리시간 T는 0.1s, 0.312s, 2s 와 같이 최대 소수점 셋째 자리까지 기록하며 뒤에는 초 단위를 의미하는 s로 끝난다.
    예를 들어, 로그 문자열 2016-09-15 03:10:33.020 0.011s은 2016년 9월 15일 오전 3시 10분 **33.010초**부터 2016년 9월 15일 오전 3시 10분 **33.020초**까지 **0.011초** 동안 처리된 요청을 의미한다. (처리시간은 시작시간과 끝시간을 포함)
    서버에는 타임아웃이 3초로 적용되어 있기 때문에 처리시간은 0.001 ≦ T ≦ 3.000이다.
    lines 배열은 응답완료시간 S를 기준으로 오름차순 정렬되어 있다.
     */
  }

  def solution(arr : Array[Int]): Array[Int]={
    val arr = Array("2016-09-15 20:59:57.421 0.351s"
      , "2016-09-15 20:59:58.233 1.181s", "2016-09-15 20:59:58.299 0.8s"
      , "2016-09-15 20:59:58.688 1.041s", "2016-09-15 20:59:59.591 1.412s"
      , "2016-09-15 21:00:00.464 1.466s", "2016-09-15 21:00:00.741 1.581s"
      , "2016-09-15 21:00:00.748 2.31s", "2016-09-15 21:00:00.966 0.381s"
      , "2016-09-15 21:00:02.066 2.62s")

    //key : 구하고자 하는 초당 최대 처리량이 변하는 순간은 단지 어떤 트래픽의 시작 또는 종료 시점뿐 입니다.
    //step1. 시작 종료로 나눠서 두배로 만들기
    //step1. 1초 단위로 쪼개서 개수 세기

    val dt = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
    dt.parse("2016-09-15 20:59:58.233").getTime

    val arr_unixtime = arr.map(v => (dt.parse(v.split(" ")(0) +" "+ v.split(" ")(1)).getTime, (v.split(" ")(2).replace("s","").toDouble*1000).toInt))

    val arr_from_to = arr_unixtime.map(v => (v._1 - v._2, v._1))

    // min max로 시작 종료 시간 구하기
    val start_min = arr_from_to.map(v=>v._1).min
    val finish_max = arr_from_to.map(v=>v._2).max

    //초단위 배열 만들기
    val second_arr = start_min to finish_max by 1000
    val arr_from_to_all = arr_from_to.map(v =>v._1) ++ arr_from_to.map(v =>v._2)
    val second_arr_cnt = second_arr.map(v => {
      (v, arr_from_to_all.filter(x => v < x && v <= x+1000 ).length)
    })

    Array(1)

  }

  def dateRef(): Unit ={

    val today = Calendar.getInstance().getTime()
    // create the date/time formatters
    val minuteFormat = new SimpleDateFormat("mm")
    val hourFormat = new SimpleDateFormat("hh")
    val amPmFormat = new SimpleDateFormat("a")
    val sssFormat = new SimpleDateFormat("SSS")
    hourFormat.format(today)
    minuteFormat.format(today)
    amPmFormat.format(today)
    sssFormat.format(today)

    val cal = Calendar.getInstance
    cal.setTime(today)
    cal.add(Calendar.SECOND, 10)
  }
}
