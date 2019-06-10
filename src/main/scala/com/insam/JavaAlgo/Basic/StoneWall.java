package com.insam.JavaAlgo.Basic;

import java.util.*;

public class StoneWall {
		/* 접근방법:
			요구한 벽을 만들기 위한 최소한의 블록수를 구한다. 최소화하기위해 블록을 최대한 재사용한다.
			이 문제는 rectilinear skyline problem, Manhattan skyline problem 유사하다.

			우선, 좌측 끝쪽 (첫번째)블록은 그대로 이용한다.
			오른쪽으로 이동하면서 이전에 사용했던 블록을 최대한 재사용하도록 한다.
			(인접한 블록에서 재사용할 수 있다는 것은 하나의 사각형으로 구성된다는 의미다.)
			스택에는 재사용 가능한 블록들이 남아있다고 보면 된다. 재사용 불가능하면 버린다.

			A. 오른쪽이 동일한 높이라면 바로 재사용
			B. 오른쪽이 더 높다면 추가로 필요한 높이의 블록을 쌓는다.
			C. 오른쪽이 더 낮다면 마지막에 올렸던 블록을 버리고 다시 A부터 반복한다.

			전형적인 탐욕 알고리즘(Greedy Algorithm﻿) 문제이다.
			즉, 그 순간에 최적이라고 생각되는 것을 선택해 나가는 방식으로 진행하여
			최종적인 해답에 도달한다.

			이렇게 해서 시간복잡도 O(N)에 이를 수 있다.
		*/
	public int solution(int[] arr) {
		int result = 0;
		
		Stack<Integer> st = new Stack<>();
		
		for(int v : arr) {
			while(!st.isEmpty() && st.peek() > v) {
				st.pop();
			}
			
			if(!st.isEmpty() && st.peek() == v) {
				continue;
			}
			
			if(st.isEmpty() || st.peek() < v) {
				result++;
				st.push(v);
			}
		}
		
		return result;
	}
}
