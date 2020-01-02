package com.insam.AlgoScala

object AlgoRun extends App {

  // 1. Iterations
  //양의정수를 받아서 바이너리열로 변환 했을 때, 가장 긴 1 사이의 갭을 구해라.
  BinaryGap.run()

  // 2. Arrays
  // 정수의 배열에서 홀수로 존재하는 정수를 찾아내라
  OddOccurencesInArray.run()

  // 정수의 배열을 K만큼 오른쪽으로 돌려라
  CycleRotation.run()

  // 3. Time Complexity
  // 1부터 시작되는 정수들의 배열에서, 딱 1개 숫자가 등장하지 않음, 그걸 찾아라
  PermMissingElem.run()

  // Tennis
  // 선수 수와 이용 가능한 코트가 주어지면 최대 병렬 테니스 게임 수를 계산하십시오.
  TennisTournament.run()

  //SocksLaundering
  //더러운 양말을 세탁기로 돌려서 깨끗한 양말과 합쳐서 최대 신 확보할 수 있는 양말세트를 찾아라
  SocksLaundering.run()

  //CountNonDivisible
  //주어진 배열에서, 각 항목들을 제거했을때 약수를 제외한 개수를 배열로 리턴해라
  CountNonDivisible.run()

  //Kakao 추석 트래픽
  

}
