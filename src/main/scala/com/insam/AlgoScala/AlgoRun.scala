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


}
